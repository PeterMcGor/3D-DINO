"""
Medical Image Segmentation Prediction Script for Meningioma Detection

This script performs inference using a trained ViTAdapterUNETR model for meningioma segmentation
from multi-channel MRI data (FLAIR + DWI + optional T2*/SWI).

Configuration Priority:
1. Command line arguments (highest priority)
2. Configuration file values (vit3d_highres.yaml)
3. Default values (lowest priority)

Required inputs:
- FLAIR: T2 FLAIR image
- DWI B1000: Diffusion-weighted image
- T2* or SWI: Optional third channel (if neither provided, uses 2-channel mode)

The script uses the exact same preprocessing pipeline as the FOMO training setup.
"""

import torch
import numpy as np
import argparse
import os
import yaml
import nibabel as nib
from functools import partial
from pathlib import Path
from scipy import ndimage

# Import your model components (adjust imports based on your project structure)
from dinov2.eval.segmentation_3d.segmentation_heads import ViTAdapterUNETRHead
from dinov2.eval.setup import setup_and_build_model_3d, get_args_parser

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, EnsureChannelFirst, Orientation, 
    Spacing, ScaleIntensityRange, CropForeground, 
    ToTensor, EnsureType, Resize, LoadImaged,
    EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd, SpatialPadd,
    ConcatItemsd, DeleteItemsd, EnsureTyped
)
import torch.nn.functional as F



def keep_largest_connected_component_3d(binary_mask):
    """
    Keep only the largest connected component in a 3D binary mask
    
    Args:
        binary_mask: 3D numpy array with 0s and 1s
    
    Returns:
        3D numpy array with only the largest connected component
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Label connected components (26-connectivity for 3D)
    labeled_array, num_features = ndimage.label(binary_mask)
    
    if num_features <= 1:
        return binary_mask
    
    # Count voxels in each component
    component_sizes = np.bincount(labeled_array.ravel())
    # Skip background (label 0)
    component_sizes[0] = 0
    
    # Find largest component
    largest_component_label = np.argmax(component_sizes)
    
    # Keep only the largest component
    largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)
    
    return largest_component_mask

def create_predict_args():
    """Create argument parser for prediction interface"""
    parser = argparse.ArgumentParser(description='Meningioma Segmentation Prediction')
    
    # Required medical imaging inputs
    parser.add_argument('--flair', type=str, required=True, 
                       help='Path to T2 FLAIR image')
    parser.add_argument('--dwi_b1000', type=str, required=True, 
                       help='Path to DWI b1000 image')
    parser.add_argument('--t2s', type=str, default=None,
                       help='Path to T2* image (optional, can be replaced with SWI or omitted for 2-channel mode)')
    parser.add_argument('--swi', type=str, default=None,
                       help='Path to SWI image (optional, can be replaced with T2* or omitted for 2-channel mode)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save segmentation NIfTI file')
    
    # Model configuration (defaults match your training configuration)
    parser.add_argument('--checkpoint-path', type=str, 
                       default='best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-file', type=str,
                       default='dinov2/configs/train/vit3d_highres.yaml',
                       help='Path to config file used during training')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                       help='Path to pretrained DinoV2 weights (overrides config if provided)')
    parser.add_argument('--arch', type=str, default='vit_large',
                       help='Model architecture (default from your training)')
    parser.add_argument('--patch-size', type=int, default=16,
                       help='Patch size (default from your training)')
    parser.add_argument('--image-size', type=int, default=112,
                       help='Image size used during training (matches global_crops_size)')
    parser.add_argument('--resize-scale', type=float, default=1.0,
                       help='Resize scale used during training')
    parser.add_argument('--segmentation-head', type=str, default='ViTAdapterUNETR',
                       help='Segmentation head type')
    parser.add_argument('--use_cls', action='store_false',
                       help='Use CLS token (for JEPA compatibility)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for sliding window inference')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap for sliding window inference')
    parser.add_argument('--cache-dir', type=str, default='./temp_cache',
                       help='Cache directory for temporary files')
    
    return parser


def load_config(config_path):
    """Load YAML configuration file and merge with defaults"""
    
    # Default configuration (from your default config)
    default_config = {
        'student': {
            'arch': 'vit_large_3d',
            'patch_size': 16,
            'drop_path_rate': 0.3,
            'layerscale': 1.0e-05,
            'drop_path_uniform': True,
            'pretrained_weights': '',
            'full_pretrained_weights': '',
            'ffn_layer': 'mlp',
            'block_chunks': 4,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True
        },
        'crops': {
            'global_crops_size': 96,
            'local_crops_size': 48
        },
        'train': {
            'batch_size_per_gpu': 128,
            'data_min_axis_size': 24,
            'OFFICIAL_EPOCH_LENGTH': 25
        },
        'optim': {
            'base_lr': 0.002
        },
        'evaluation': {
            'eval_period_iterations': 12500
        }
    }
    
    # Load and merge highres config if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            highres_config = yaml.safe_load(f)
        
        # Merge configurations (highres overrides default)
        def merge_configs(default, override):
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_configs(default_config, highres_config)
        print(f"Loaded config from: {config_path}")
    else:
        config = default_config
        print(f"Warning: Config file {config_path} not found. Using defaults.")
    
    return config


def prepare_input_images(args):
    """Prepare input images using the exact FOMO transforms from training"""
    
    # Load reference image to get original dimensions and affine
    reference_img = nib.load(args.flair)
    original_shape = reference_img.shape
    original_affine = reference_img.affine
    
    print(f"Original image shape: {original_shape}")
    print(f"Processing with image size: {args.image_size}")
    
    # Determine number of channels and setup data dictionary
    if args.t2s is not None and args.swi is not None:
        print("Warning: Both T2* and SWI provided. Using T2*.")
        data_dict = {
            "image1": args.flair,
            "image2": args.dwi_b1000,
            "image3": args.t2s
        }
        dataset_type = "3channels"
        print("Using 3-channel mode: FLAIR + DWI + T2*")
    elif args.t2s is not None:
        data_dict = {
            "image1": args.flair,
            "image2": args.dwi_b1000,
            "image3": args.t2s
        }
        dataset_type = "3channels"
        print("Using 3-channel mode: FLAIR + DWI + T2*")
    elif args.swi is not None:
        data_dict = {
            "image1": args.flair,
            "image2": args.dwi_b1000,
            "image3": args.swi
        }
        dataset_type = "3channels"
        print("Using 3-channel mode: FLAIR + DWI + SWI")
    else:
        # 2-channel mode: only FLAIR + DWI
        data_dict = {
            "image1": args.flair,
            "image2": args.dwi_b1000
        }
        dataset_type = "2channels"
        print("Using 2-channel mode: FLAIR + DWI")
    
    # Create transforms based on the exact FOMO training transforms
    if dataset_type == "3channels":
        transforms = Compose([
            # Load 3 Nifti images (ensure_channel_first=True already handles channels)
            LoadImaged(keys=["image1", "image2", "image3"], ensure_channel_first=True),
            # Concatenate the 3 images into a single multi-channel image
            ConcatItemsd(keys=["image1", "image2", "image3"], name='image', dim=0),
            # Remove the original individual image keys
            DeleteItemsd(keys=["image1", "image2", "image3"]),
            # Ensure proper tensor types
            EnsureTyped(keys=["image"]),
            # Spatial orientation
            Orientationd(keys=["image"], axcodes="RAS"),
            # Resample to target spacing (1mm isotropic)
            Spacingd(
                keys=["image"],
                pixdim=(1.0 / args.resize_scale, 1.0 / args.resize_scale, 1.0 / args.resize_scale),
                mode="bilinear",
            ),
            # Intensity normalization (per channel for the 3-channel image)
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=99.95, b_min=-1.0, b_max=1.0, 
                clip=True, channel_wise=True
            ),
            # Spatial padding to ensure minimum size
            SpatialPadd(keys=["image"], spatial_size=(args.image_size, args.image_size, args.image_size), value=-1.0),
        ])
    else:  # 2channels
        transforms = Compose([
            # Load 2 Nifti images
            LoadImaged(keys=["image1", "image2"], ensure_channel_first=True),
            # Concatenate the 2 images into a single multi-channel image
            ConcatItemsd(keys=["image1", "image2"], name='image', dim=0),
            # Remove the original individual image keys
            DeleteItemsd(keys=["image1", "image2"]),
            # Ensure proper tensor types
            EnsureTyped(keys=["image"]),
            # Spatial orientation
            Orientationd(keys=["image"], axcodes="RAS"),
            # Resample to target spacing (1mm isotropic)
            Spacingd(
                keys=["image"],
                pixdim=(1.0 / args.resize_scale, 1.0 / args.resize_scale, 1.0 / args.resize_scale),
                mode="bilinear",
            ),
            # Intensity normalization (per channel for the 2-channel image)
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=99.95, b_min=-1.0, b_max=1.0, 
                clip=True, channel_wise=True
            ),
            # Spatial padding to ensure minimum size
            SpatialPadd(keys=["image"], spatial_size=(args.image_size, args.image_size, args.image_size), value=-1.0),
        ])
    
    # Apply transforms
    processed_data = transforms(data_dict)
    
    # Get the processed image tensor
    input_tensor = processed_data["image"]
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    return input_tensor, reference_img, dataset_type


def load_segmentation_model(args, input_channels, config):
    """Load the trained segmentation model"""
    
    print("Loading feature model...")
    
    # Parameter priority: Command line args > Config file > Defaults
    # Extract model parameters from config, with command-line overrides
    
    # Architecture: Use command line if provided, otherwise use config
    if hasattr(args, 'arch') and args.arch:
        model_arch = args.arch
        print(f"Using architecture from command line: {model_arch}")
    else:
        model_arch = config.get('student', {}).get('arch', 'vit_large_3d')
        print(f"Using architecture from config: {model_arch}")
    
    # Patch size: Use command line if provided, otherwise use config
    if hasattr(args, 'patch_size') and args.patch_size:
        patch_size = args.patch_size
        print(f"Using patch size from command line: {patch_size}")
    else:
        patch_size = config.get('student', {}).get('patch_size', 16)
        print(f"Using patch size from config: {patch_size}")
    
    # Pretrained weights: Use command line if provided, otherwise use config
    if args.pretrained_weights:
        pretrained_weights = args.pretrained_weights
        print(f"Using pretrained weights from command line: {pretrained_weights}")
    else:
        pretrained_weights = config.get('student', {}).get('full_pretrained_weights', '')
        print(f"Using pretrained weights from config: {pretrained_weights}")
    
    # Handle arch mapping (vit_large_3d -> vit_large for compatibility)
    if model_arch == 'vit_large_3d':
        model_arch = 'vit_large'
        print(f"Mapped architecture to: {model_arch}")
    
    print(f"\nFinal configuration:")
    print(f"  - Model architecture: {model_arch}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Image size: {args.image_size}")
    print(f"  - Pretrained weights: {pretrained_weights}")
    print(f"  - Input channels: {input_channels}")
    print(f"  - Segmentation head: {args.segmentation_head}")
    
    # Create a complete args object using the actual parser and override values
    model_args = get_args_parser(add_help=False).parse_args([])
    
    # Override with our values
    model_args.config_file = args.config_file
    model_args.pretrained_weights = pretrained_weights
    model_args.arch = model_arch
    model_args.patch_size = patch_size
    model_args.output_dir = './temp_output'
    model_args.train_dataset_str = ''
    model_args.val_dataset_str = ''
    
    # Add missing attributes required by get_cfg_from_args_3d
    model_args.cache_dir = getattr(args, 'cache_dir', './temp_cache')
    model_args.opts = []  # Empty list for additional config options
    model_args.jepa_learning = False  # Use DINO config by default
    
    # Ensure temporary directories exist
    os.makedirs('./temp_output', exist_ok=True)
    os.makedirs(model_args.cache_dir, exist_ok=True)
    
    feature_model, autocast_dtype = setup_and_build_model_3d(model_args)
    
    # Create segmentation model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    
    num_classes = 2     # Background + Meningioma
    
    if args.segmentation_head == 'ViTAdapterUNETR':
        seg_model = ViTAdapterUNETRHead(
            feature_model, 
            input_channels, 
            args.image_size, 
            num_classes, 
            autocast_ctx, 
            use_cls=args.use_cls
        )
    else:
        raise ValueError(f"Segmentation head {args.segmentation_head} not supported")
    
    # Load trained weights
    print(f"\nLoading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    seg_model.load_state_dict(checkpoint)
    
    seg_model.cuda()
    seg_model.eval()
    
    print("Model loaded successfully!")
    return seg_model


def run_prediction(model, input_tensor, args):
    """Run segmentation prediction"""
    
    print("Running inference...")
    
    with torch.no_grad():
        input_tensor = input_tensor.cuda()
        
        # Run sliding window inference
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=(args.image_size,) * 3,
            sw_batch_size=args.batch_size,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian"
        )
        
        # Apply softmax and get binary mask
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1, keepdim=True)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()


def postprocess_and_save(predictions, reference_img, output_path):
    """Postprocess predictions and save to original image space"""
    
    # Remove batch and channel dimensions
    pred_mask = predictions.squeeze()
    
    # Resize back to original dimensions
    original_shape = reference_img.shape
    
    print(f"Resizing prediction from {pred_mask.shape} to {original_shape}")
    
    # Convert to tensor for interpolation
    pred_tensor = torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0)
    
    # Resize to original dimensions
    resized_pred = F.interpolate(
        pred_tensor, 
        size=original_shape, 
        mode='nearest'
    ).squeeze().numpy()
    
    # Ensure binary mask
    binary_mask = (resized_pred > 0.5).astype(np.uint8)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    binary_mask = keep_largest_connected_component_3d(binary_mask)
    
    # Save with original affine and header
    output_img = nib.Nifti1Image(
        binary_mask.astype(np.uint8), 
        reference_img.affine, 
        reference_img.header
    )
    
    nib.save(output_img, output_path)
    print(f"Binary segmentation saved to: {output_path}")
    
    # Print some statistics
    num_voxels = np.sum(binary_mask)
    total_voxels = np.prod(binary_mask.shape)
    percentage = (num_voxels / total_voxels) * 100
    
    print(f"Segmentation statistics:")
    print(f"  - Segmented voxels: {num_voxels}")
    print(f"  - Total voxels: {total_voxels}")
    print(f"  - Percentage segmented: {percentage:.2f}%")


def main():
    """Main prediction function"""
    
    # Parse arguments
    parser = create_predict_args()
    args = parser.parse_args()
    
    print("=== Meningioma Segmentation Prediction ===")
    print(f"FLAIR: {args.flair}")
    print(f"DWI B1000: {args.dwi_b1000}")
    print(f"T2*: {args.t2s}")
    print(f"SWI: {args.swi}")
    print(f"Output: {args.output}")
    print()
    
    # Load configuration and update args with config values if needed
    print("Loading configuration...")
    config = load_config(args.config_file)
    
    # Update image size from config if not explicitly set
    if args.image_size == 112:  # default value
        config_image_size = config.get('crops', {}).get('global_crops_size', 112)
        if config_image_size != 112:
            args.image_size = config_image_size
            print(f"Updated image size from config: {args.image_size}")
    
    print(f"Using image size: {args.image_size}")
    print(f"Using config file: {args.config_file}")
    
    # Validate inputs
    if not os.path.exists(args.flair):
        raise FileNotFoundError(f"FLAIR image not found: {args.flair}")
    if not os.path.exists(args.dwi_b1000):
        raise FileNotFoundError(f"DWI image not found: {args.dwi_b1000}")
    
    # Check third channel availability (both can be None for 2-channel mode)
    if args.t2s is not None and not os.path.exists(args.t2s):
        raise FileNotFoundError(f"T2* image not found: {args.t2s}")
    if args.swi is not None and not os.path.exists(args.swi):
        raise FileNotFoundError(f"SWI image not found: {args.swi}")
    
    # If neither T2* nor SWI is provided, we'll use 2-channel mode
    if args.t2s is None and args.swi is None:
        print("Neither T2* nor SWI provided. Using 2-channel mode (FLAIR + DWI only).")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    try:
        # Step 1: Prepare input images and determine input channels
        print("Step 1: Preparing input images...")
        input_tensor, reference_img, dataset_type = prepare_input_images(args)
        
        # Determine input channels based on dataset type
        input_channels = 3 if dataset_type == "3channels" else 2
        print(f"Using {input_channels}-channel input")
        
        # Step 2: Load model
        print("Step 2: Loading segmentation model...")
        model = load_segmentation_model(args, input_channels, config)
        
        # Step 3: Run prediction
        print("Step 3: Running prediction...")
        predictions, probabilities = run_prediction(model, input_tensor, args)
        
        # Step 4: Postprocess and save
        print("Step 4: Postprocessing and saving results...")
        postprocess_and_save(predictions, reference_img, args.output)
        
        print("\n=== Prediction completed successfully! ===")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    main()