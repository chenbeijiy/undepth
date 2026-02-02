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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.gamma = 1.0
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        # Improvement 3.3: Multi-loss joint optimization
        # L_converge_enhanced = λ1 * L_converge_local + λ2 * L_converge_global + λ3 * L_cross
        self.lambda_converge_local = 7.0  # Local convergence loss (original adjacent constraint)
        # self.lambda_converge_global = 2.0  # Global convergence loss (Improvement 2.1)
        # self.lambda_converge_cross = 1.0  # Depth-Alpha cross term (Improvement 2.5)
        
        # Keep for backward compatibility
        self.lambda_converge = self.lambda_converge_local
        
        # Improvement 3.3: Multi-loss joint optimization parameters
        # self.lambda_multiview_depth = 0.5  # Multi-view depth consistency loss weight
        # self.lambda_alpha_completeness = 0.1  # Alpha completeness loss weight
        self.opacity_cull = 0.05
        
        # DISABLED: Innovation 6: Adaptive Densification for Holes
        # self.adaptive_densify_enabled = False # Enable adaptive densification for holes
        # self.adaptive_densify_depth_var_threshold = 0.01  # Depth variance threshold for hole detection
        # self.adaptive_densify_alpha_threshold = 0.5  # Alpha threshold for hole detection
        # self.adaptive_densify_max_gaussians = 1000  # Maximum number of Gaussians to add per iteration
        # self.adaptive_densify_interval = 500  # Interval for adaptive densification (iterations)
        
        # DISABLED: Innovation 3: Adaptive Alpha Enhancement
        # self.adaptive_alpha_enhance_enabled = False  # Enable adaptive alpha enhancement
        # self.lambda_alpha_enhance = 0.2  # Alpha enhancement loss weight
        # self.alpha_enhance_from_iter = 5000  # Start iteration for alpha enhancement
        # self.alpha_enhance_depth_var_scale = 10.0  # Depth variance scale factor for hole detection sensitivity
        
        # DISABLED: Innovation 1: Spatial-Depth Coherence Loss
        # self.spatial_depth_coherence_enabled = False  # Enable spatial-depth coherence loss (default: enabled)
        # self.lambda_spatial_depth = 0.1  # Spatial-depth coherence loss weight
        # self.spatial_depth_from_iter = 0  # Start iteration for spatial-depth coherence (0 = from start)
        # self.spatial_depth_rgb_weight = 0.1  # RGB similarity weight coefficient (lambda in exp(-lambda * ||I - I'||^2))
        # self.spatial_depth_kernel_size = 3  # Neighborhood kernel size (3 or 5)
        
        # DISABLED: Innovation 4: Depth-Normal Joint Optimization
        # self.depth_normal_consistency_enabled = False  # Enable depth-normal consistency loss (default: enabled)
        # self.lambda_depth_normal = 0.1  # Depth-normal consistency loss weight
        # self.depth_normal_from_iter = 0  # Start iteration for depth-normal consistency (0 = from start)
        
        # DISABLED: Improvement 2.3: Spatial Depth Smoothness Loss
        # self.spatial_depth_smoothness_enabled = True  # Enable spatial depth smoothness loss (default: enabled)
        # self.lambda_spatial_smooth = 0.1  # Spatial depth smoothness loss weight
        # self.spatial_smooth_from_iter = 0  # Start iteration for spatial smoothness (0 = from start)

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter      = 500
        self.densify_until_iter     = 20000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
