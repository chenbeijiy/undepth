#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from random import randint
import time
from datetime import timedelta

import json
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import uuid
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset: ModelParams,
             opt:     OptimizationParams,
             pipe:    PipelineParams,
             testing_iterations,
             saving_iterations,
             checkpoint_iterations,
             checkpoint,
             logger_enabled):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, logger_enabled)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)  ## load data
    gaussians.training_setup(opt)
    scene.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # Record training start time for total training time calculation
    training_start_time = time.time()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_converge_for_log = 0.0
    ema_alpha_concentration_for_log = 0.0
    ema_alpha_completeness_for_log = 0.0
    ema_multiview_depth_for_log = 0.0
    # Improvement 3.1: Enhanced depth convergence loss combination
    ema_converge_local_for_log = 0.0
    ema_converge_global_for_log = 0.0
    ema_converge_cross_for_log = 0.0
    # DISABLED: All improvements and innovations
    # ema_alpha_enhance_for_log = 0.0
    # ema_spatial_depth_for_log = 0.0
    # ema_depth_normal_for_log = 0.0
    # ema_spatial_smooth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image                  = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter      = render_pkg["visibility_filter"]
        radii                  = render_pkg["radii"]
        converge               = render_pkg["converge"]

        # Gamma corrected Image
        gt_image = viewpoint_cam.original_image.cuda().pow(dataset.gamma)
        gt_image.pow(dataset.gamma)

        ssim_value = ssim(image, gt_image)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # Improvement 3.3: Multi-loss joint optimization
        # L_converge_enhanced = λ1 * L_converge_local + λ2 * L_converge_global + λ3 * L_cross
        
        # Local convergence loss (original adjacent constraint)
        lambda_converge_local = opt.lambda_converge_local if iteration > 10000 else 0.00
        converge_local_loss = lambda_converge_local * converge.mean()
        
        # Global convergence loss (Improvement 2.1)
        # lambda_converge_global = opt.lambda_converge_global if iteration > 10000 else 0.0
        # converge_ray = render_pkg.get('converge_ray', torch.tensor(0.0, device="cuda"))
        # converge_global_loss = lambda_converge_global * converge_ray.mean()
        
        # Depth-Alpha cross term (Improvement 2.5)
        # lambda_converge_cross = opt.lambda_converge_cross if iteration > 10000 else 0.0
        # depth_alpha_cross = render_pkg.get('depth_alpha_cross', torch.tensor(0.0, device="cuda"))
        # converge_cross_loss = lambda_converge_cross * depth_alpha_cross.mean()
        
        # Combined enhanced convergence loss
        converge_enhanced = converge_local_loss

        rend_dist   = render_pkg["rend_dist"]  
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Improvement 3.3: Alpha completeness loss
        # lambda_alpha_completeness = opt.lambda_alpha_completeness if iteration > 8000 else 0.0
        # rend_alpha = render_pkg['rend_alpha']
        # Use converge_ray as a proxy for depth variance (low converge_ray means low variance, i.e., valid surface)
        # converge_ray represents depth variance, so low values indicate good convergence (valid surface)
        # valid_surface_mask = (converge_ray < 0.1).float()
        # alpha_completeness_loss = lambda_alpha_completeness * ((1.0 - rend_alpha) ** 2 * valid_surface_mask).mean()

        # DISABLED: Innovation 3: Adaptive Alpha Enhancement
        alpha_enhance_loss = torch.tensor(0.0, device="cuda")
        # if opt.adaptive_alpha_enhance_enabled:
        #     lambda_alpha_enhance = opt.lambda_alpha_enhance if iteration > opt.alpha_enhance_from_iter else 0.0
        #     if lambda_alpha_enhance > 0:
        #         from utils.alpha_enhancement import adaptive_alpha_enhancement_loss
        #         rend_alpha = render_pkg['rend_alpha']
        #         surf_depth = render_pkg['surf_depth']
        #         alpha_enhance_loss = adaptive_alpha_enhancement_loss(
        #             rend_alpha=rend_alpha,
        #             surf_depth=surf_depth,
        #             lambda_enhance=lambda_alpha_enhance,
        #             depth_var_scale=opt.alpha_enhance_depth_var_scale
        #         )

        # DISABLED: Innovation 1: Spatial-Depth Coherence Loss
        spatial_depth_loss = torch.tensor(0.0, device="cuda")
        # if opt.spatial_depth_coherence_enabled:
        #     lambda_spatial = opt.lambda_spatial_depth if iteration > opt.spatial_depth_from_iter else 0.0
        #     if lambda_spatial > 0:
        #         from utils.spatial_depth_coherence import spatial_depth_coherence_loss
        #         surf_depth = render_pkg['surf_depth']
        #         rendered_rgb = render_pkg['render']  # (3, H, W)
        #         spatial_depth_loss = spatial_depth_coherence_loss(
        #             surf_depth=surf_depth,
        #             rgb=rendered_rgb,
        #             lambda_spatial=opt.spatial_depth_rgb_weight,
        #             kernel_size=opt.spatial_depth_kernel_size
        #         ) * lambda_spatial

        # DISABLED: Innovation 4: Depth-Normal Joint Optimization
        depth_normal_loss = torch.tensor(0.0, device="cuda")
        # if opt.depth_normal_consistency_enabled:
        #     lambda_dn = opt.lambda_depth_normal if iteration > opt.depth_normal_from_iter else 0.0
        #     if lambda_dn > 0:
        #         from utils.depth_normal_consistency import depth_normal_consistency_loss
        #         surf_depth = render_pkg['surf_depth']
        #         surf_normal = render_pkg['surf_normal']
        #         depth_normal_loss = depth_normal_consistency_loss(
        #             surf_depth=surf_depth,
        #             surf_normal=surf_normal,
        #             lambda_dn=lambda_dn
        #         )

        # DISABLED: Improvement 2.3: Spatial Depth Smoothness Loss
        spatial_smooth_loss = torch.tensor(0.0, device="cuda")
        # if opt.spatial_depth_smoothness_enabled:
        #     lambda_smooth = opt.lambda_spatial_smooth if iteration > opt.spatial_smooth_from_iter else 0.0
        #     if lambda_smooth > 0:
        #         from utils.spatial_depth_smoothness import spatial_depth_smoothness_loss
        #         surf_depth = render_pkg['surf_depth']
        #         spatial_smooth_loss = spatial_depth_smoothness_loss(
        #             surf_depth=surf_depth,
        #             lambda_smooth=lambda_smooth
        #         )

        total_loss = loss + dist_loss + normal_loss + converge_enhanced
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_converge_for_log = 0.4 * converge_enhanced.item() + 0.6 * ema_converge_for_log
            ema_converge_local_for_log = 0.4 * converge_local_loss.item() + 0.6 * ema_converge_local_for_log
            # ema_converge_global_for_log = 0.4 * converge_global_loss.item() + 0.6 * ema_converge_global_for_log
            # ema_converge_cross_for_log = 0.4 * converge_cross_loss.item() + 0.6 * ema_converge_cross_for_log
            # ema_alpha_completeness_for_log = 0.4 * alpha_completeness_loss.item() + 0.6 * ema_alpha_completeness_for_log
            # DISABLED: All improvements and innovations
            # ema_alpha_enhance_for_log = 0.4 * alpha_enhance_loss.item() + 0.6 * ema_alpha_enhance_for_log
            # ema_spatial_depth_for_log = 0.4 * spatial_depth_loss.item() + 0.6 * ema_spatial_depth_for_log
            # ema_depth_normal_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_depth_normal_for_log
            # ema_spatial_smooth_for_log = 0.4 * spatial_smooth_loss.item() + 0.6 * ema_spatial_smooth_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "converge": f"{ema_converge_for_log:.{5}f}",
                    # "conv_loc": f"{ema_converge_local_for_log:.{5}f}",  # Improvement 3.3: Local
                    # "conv_glob": f"{ema_converge_global_for_log:.{5}f}",  # Improvement 3.3: Global
                    # "conv_cross": f"{ema_converge_cross_for_log:.{5}f}",  # Improvement 3.3: Cross
                    # "alpha_comp": f"{ema_alpha_completeness_for_log:.{5}f}",  # Improvement 3.3: Alpha completeness
                    # "alpha_enh": f"{ema_alpha_enhance_for_log:.{5}f}",  # Innovation 3: Adaptive Alpha Enhancement
                    # "spat_depth": f"{ema_spatial_depth_for_log:.{5}f}",  # Innovation 1: Spatial-Depth Coherence
                    # "depth_norm": f"{ema_depth_normal_for_log:.{5}f}",  # Innovation 4: Depth-Normal Consistency
                    "Points": f"{len(gaussians.get_xyz)}"  
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)

            # Calculate total training time if this is the last iteration
            total_training_time_for_report = None
            if iteration == opt.iterations:
                total_training_time_for_report = time.time() - training_start_time
                progress_bar.close()
                
                # Format total training time using timedelta (simpler)
                time_delta = timedelta(seconds=int(total_training_time_for_report))
                
                # Log total training time to TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar('training/total_time_seconds', total_training_time_for_report, iteration)
                    tb_writer.add_scalar('training/total_time_hours', total_training_time_for_report / 3600.0, iteration)
                
                # Print total training time (timedelta automatically formats nicely)
                print(f"\n[Training Complete] Total training time: {time_delta} ({total_training_time_for_report:.2f} seconds)")

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), dataset, total_training_time_for_report)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # DISABLED: Innovation 6: Adaptive Densification for Holes
                # if opt.adaptive_densify_enabled and iteration > opt.densify_from_iter:
                #     if iteration % opt.adaptive_densify_interval == 0:
                #         from utils.adaptive_densification import adaptive_densification_for_holes
                #         try:
                #             num_added = adaptive_densification_for_holes(
                #                 gaussians=gaussians,
                #                 render_pkg=render_pkg,
                #                 viewpoint=viewpoint_cam,
                #                 depth_var_threshold=opt.adaptive_densify_depth_var_threshold,
                #                 alpha_threshold=opt.adaptive_densify_alpha_threshold,
                #                 max_new_gaussians=opt.adaptive_densify_max_gaussians,
                #                 scene_extent=scene.cameras_extent
                #             )
                #             if num_added > 0:
                #                 print(f"[ITER {iteration}] Adaptive densification: Added {num_added} Gaussians in hole regions")
                #         except Exception as e:
                #             print(f"[ITER {iteration}] Warning: Adaptive densification failed: {e}")

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Maximum size limit
            if iteration >= opt.densify_until_iter:
                gaussians.clamp_scaling(torch.tensor(0.1 * scene.cameras_extent).cuda())

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args, logger_enabled):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and logger_enabled:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(
    tb_writer, iteration,
    Ll1, loss, l1_loss,
    elapsed,
    testing_iterations,
    scene : Scene,
    renderFunc,
    renderArgs,
    dataset: ModelParams,
    total_training_time=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        data = {'test': {}, 'train': {}}

        # Select cameras for training and testing
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # Traverse cameras
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                    image = torch.clamp(render_pkg["render"].pow(1.0 / dataset.gamma), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                data[config['name']][f'{iteration}_psnr'] = psnr_test.item()
                data[config['name']][f'{iteration}_l1'] = l1_test.item()
        
        # Add total training time to output.json if available
        if total_training_time is not None:
            time_delta = timedelta(seconds=int(total_training_time))
            data['training'] = {
                'total_time_seconds': total_training_time,
                'total_time_hours': total_training_time / 3600.0,
                'total_time_formatted': str(time_delta)
            }
        
        # Write to output.json
        with open(os.path.join(scene.model_path, "output.json"), 'w') as file:
            json.dump(data, file, indent=2)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000])  ##default=[7_000, 30_000]
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 0)
    parser.add_argument("--logger_enabled", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.logger_enabled)

    # All done
    print("\nTraining complete.")