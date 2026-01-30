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

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_converge_for_log = 0.0
    ema_alpha_concentration_for_log = 0.0
    ema_alpha_completeness_for_log = 0.0
    ema_converge_ray_for_log = 0.0
    ema_multiview_depth_for_log = 0.0
    # DISABLED: Improvement 2.5
    # ema_depth_alpha_cross_for_log = 0.0

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

        # Converge Loss - Original adjacent constraint
        lambda_converge = opt.lambda_converge if iteration > 10000 else 0.00
        converge_loss = lambda_converge * converge.mean()
        
        # DISABLED: Improvement 2.1: Global depth convergence loss
        # lambda_converge_ray = opt.lambda_converge_ray if iteration > 10000 else 0.0
        # converge_ray = render_pkg.get('converge_ray', torch.tensor(0.0, device="cuda"))
        # converge_ray_loss = lambda_converge_ray * converge_ray.mean()
        converge_ray_loss = torch.tensor(0.0, device="cuda")

        rend_dist   = render_pkg["rend_dist"]  
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # DISABLED: Improvement 2.3: Alpha concentration and completeness losses
        # lambda_alpha_concentration = opt.lambda_alpha_concentration if iteration > 8000 else 0.0
        # lambda_alpha_completeness = opt.lambda_alpha_completeness if iteration > 8000 else 0.0
        # alpha_concentration = render_pkg.get('alpha_concentration', torch.tensor(0.0, device="cuda"))
        # alpha_concentration_loss = lambda_alpha_concentration * alpha_concentration.mean()
        # rend_alpha = render_pkg['rend_alpha']
        # depth_variance = render_pkg.get('depth_variance', torch.tensor(1.0, device="cuda"))
        # valid_surface_mask = (depth_variance < 0.1).float()
        # alpha_completeness_loss = lambda_alpha_completeness * ((1.0 - rend_alpha) ** 2 * valid_surface_mask).mean()
        alpha_concentration_loss = torch.tensor(0.0, device="cuda")
        alpha_completeness_loss = torch.tensor(0.0, device="cuda")

        # DISABLED: Improvement 2.4: Multi-view depth consistency loss
        # lambda_multiview_depth = opt.lambda_multiview_depth if iteration > 12000 else 0.0
        # multiview_depth_loss = torch.tensor(0.0, device="cuda")
        # 
        # if lambda_multiview_depth > 0 and len(viewpoint_stack) > 0:
        #     # Sample another random view for multi-view consistency
        #     other_view_idx = randint(0, len(viewpoint_stack) - 1)
        #     other_viewpoint_cam = viewpoint_stack[other_view_idx]
        #     
        #     # Render the other view (need gradients for loss computation)
        #     other_render_pkg = render(other_viewpoint_cam, gaussians, pipe, background)
        #     other_surf_depth = other_render_pkg['surf_depth']
        #     
        #     # Compute multi-view depth consistency loss
        #     from utils.multiview_utils import multi_view_depth_consistency_loss
        #     surf_depth = render_pkg['surf_depth']
        #     multiview_depth_loss = lambda_multiview_depth * multi_view_depth_consistency_loss(
        #         surf_depth, viewpoint_cam, other_surf_depth, other_viewpoint_cam
        #     )
        multiview_depth_loss = torch.tensor(0.0, device="cuda")
        
        # DISABLED: Improvement 2.5: Depth-Alpha joint optimization
        # lambda_depth_alpha_cross = opt.lambda_depth_alpha_cross if iteration > 10000 else 0.0
        # depth_alpha_cross_loss = torch.tensor(0.0, device="cuda")
        # 
        # if lambda_depth_alpha_cross > 0:
        #     # Extract depth-alpha cross term from render package
        #     depth_alpha_cross = render_pkg.get('depth_alpha_cross', torch.tensor(0.0, device="cuda"))
        #     depth_alpha_cross_loss = lambda_depth_alpha_cross * depth_alpha_cross.mean()
        # else:
        #     depth_alpha_cross_loss = torch.tensor(0.0, device="cuda")
        depth_alpha_cross_loss = torch.tensor(0.0, device="cuda")

        # loss
        total_loss = loss + dist_loss + normal_loss + converge_loss + converge_ray_loss + multiview_depth_loss + depth_alpha_cross_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_converge_for_log = 0.4 * converge_loss.item() + 0.6 * ema_converge_for_log
            # DISABLED: Improvement 2.4
            # ema_multiview_depth_for_log = 0.4 * multiview_depth_loss.item() + 0.6 * ema_multiview_depth_for_log
            # DISABLED: Improvement 2.1
            # ema_converge_ray_for_log = 0.4 * converge_ray_loss.item() + 0.6 * ema_converge_ray_for_log
            # DISABLED: Improvements 2.2 & 2.3
            # ema_alpha_concentration_for_log = 0.4 * alpha_concentration_loss.item() + 0.6 * ema_alpha_concentration_for_log
            # ema_alpha_completeness_for_log = 0.4 * alpha_completeness_loss.item() + 0.6 * ema_alpha_completeness_for_log
            # DISABLED: Improvement 2.5
            # ema_depth_alpha_cross_for_log = 0.4 * depth_alpha_cross_loss.item() + 0.6 * ema_depth_alpha_cross_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "converge": f"{ema_converge_for_log:.{5}f}",
                    # DISABLED: Improvement 2.5
                    # "depth_alpha": f"{ema_depth_alpha_cross_for_log:.{5}f}",
                    # DISABLED: Improvement 2.4
                    # "multiview": f"{ema_multiview_depth_for_log:.{5}f}",
                    # DISABLED: Improvement 2.1
                    # "converge_ray": f"{ema_converge_ray_for_log:.{5}f}",
                    # DISABLED: Improvements 2.2 & 2.3
                    # "alpha_conc": f"{ema_alpha_concentration_for_log:.{5}f}",
                    # "alpha_comp": f"{ema_alpha_completeness_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"  
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), dataset)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # Improvement 2.7: Adaptive densification based on depth variance
                if opt.adaptive_densify_enabled and iteration > opt.densify_from_iter:
                    depth_variance = render_pkg.get('depth_variance', None)
                    if depth_variance is not None:
                        # Detect depth dispersion regions (holes risk areas)
                        dispersion_mask = depth_variance.squeeze() > opt.depth_variance_threshold  # (H, W)
                        
                        if dispersion_mask.any():
                            # Get 2D positions of visible Gaussians
                            # viewspace_point_tensor is in NDC coordinates [-1, 1], convert to pixel coordinates
                            means2D = viewspace_point_tensor.detach()
                            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
                            
                            # Convert NDC coordinates to pixel coordinates
                            # NDC: [-1, 1] -> Pixel: [0, W-1] and [0, H-1]
                            pixel_coords = torch.zeros_like(means2D[:, :2])
                            pixel_coords[:, 0] = (means2D[:, 0] + 1.0) * (W - 1) / 2.0  # x coordinate
                            pixel_coords[:, 1] = (means2D[:, 1] + 1.0) * (H - 1) / 2.0  # y coordinate
                            pixel_coords = pixel_coords.long()  # (N, 2)
                            
                            # Clamp to valid image bounds
                            pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, W - 1)
                            pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, H - 1)
                            
                            # Check which Gaussians are in dispersion regions
                            dispersion_at_gaussians = dispersion_mask[pixel_coords[:, 1], pixel_coords[:, 0]]  # (N,)
                            
                            # Combine with visibility filter
                            adaptive_densify_filter = visibility_filter & dispersion_at_gaussians
                            
                            # Add densification stats for Gaussians in dispersion regions
                            # Increase gradient accumulation to encourage densification
                            if adaptive_densify_filter.any():
                                # Add extra gradient accumulation for adaptive densification
                                extra_grad = torch.norm(viewspace_point_tensor.grad[adaptive_densify_filter], dim=-1, keepdim=True)
                                gaussians.xyz_gradient_accum[adaptive_densify_filter] += extra_grad * 2.0  # Boost by 2x
                                gaussians.denom[adaptive_densify_filter] += 1

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
    dataset: ModelParams):
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
        
        # Write to output.json
        with open(os.path.join(scene.model_path, "output.json"), 'w') as file:
            json.dump(data, file)

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