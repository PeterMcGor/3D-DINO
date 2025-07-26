#!/usr/bin/env python3
"""
Script to generate YAML job files for cross-validation training
Uses the working conda environment setup pattern
"""

import os
import argparse
from pathlib import Path

def generate_fold_yaml(fold_number, total_folds=5, github_username="PeterMcGor", branch_name="fomo-finetuning-t2"):
    """Generate YAML content for a specific fold using the working conda setup"""
    
    yaml_content = f"""schema_version: 2024.04.01
type: job
spec:
  name: dino3d-fomo-fold{fold_number}-training-499
  owner: dgm-ms-brain-mri/pedro-maciasgordaliza
  description: '3D-DINO finetuning on FOMO dataset fold {fold_number}'
  image: nvidia-oci/saturncloud/saturn-python-pytorch:2024.08.01
  instance_type: 1xA100
  environment_variables:
    FOLD_NUMBER: "{fold_number}"
    PYTHONPATH: "/home/jovyan/workspace/3D-DINO"
    BASE_DATA_DIR: "/home/jovyan/workspace/3D-DINO/dino_datasets/fomo-task2_2channels/"
    OUTPUT_DIR: "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/499_fold_{fold_number}"
    CACHE_DIR: "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/499_cache_dir_fold_{fold_number}"
    PRETRAINED_WEIGHTS: "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/Dino3d_last-models/highres_teacher_checkpoint.pth"
    DATASET_NAME: "fomo-task2_2channels_{fold_number}"
  working_directory: /home/jovyan/workspace/3D-DINO
  token_scope: job:{{self}}:dask:write
  git_repositories:
    - url: git@github.com:{github_username}/3D-DINO.git
      path: /home/jovyan/workspace/3D-DINO
      public: false
  secrets: []
  shared_folders:
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/fomo25
      name: fomo25
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset
      name: openmind-dataset
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/ms-data
      name: ms-data
  command: |
    bash -c "
    echo '=================================================' &&
    echo 'DINO Training Job - Fold {fold_number}' &&
    echo '=================================================' &&
    
    echo 'Step 1: Setup git repository...' &&
    git fetch origin {branch_name} &&
    git checkout {branch_name} &&
    git reset --hard origin/{branch_name} &&
    
    echo 'Step 2: Create conda environment...' &&
    conda create -n dino3d python=3.9 -y &&
    
    echo 'Step 3: Initialize and activate environment...' &&
    conda init bash &&
    source ~/.bashrc &&
    conda activate dino3d &&
    
    echo 'Step 4: Install requirements...' &&
    pip install -r requirements.txt --quiet &&
    
    echo 'Step 5: Create output directories...' &&
    mkdir -p $OUTPUT_DIR &&
    mkdir -p $CACHE_DIR &&
    
    echo 'Step 6: Verify environment...' &&
    echo 'Current conda environment:' $CONDA_DEFAULT_ENV &&
    python --version &&
    python -c 'import torch; print(\\\"PyTorch version:\\\", torch.__version__); print(\\\"CUDA available:\\\", torch.cuda.is_available())' &&
    
    echo 'Step 7: Starting DINO training - Fold {fold_number}...' &&
    echo 'Training parameters:' &&
    echo '  - Fold: {fold_number}' &&
    echo '  - Dataset: $DATASET_NAME' &&
    echo '  - Output dir: $OUTPUT_DIR' &&
    echo '  - Cache dir: $CACHE_DIR' &&
    
    python dinov2/eval/segmentation3d.py \\
      --config-file 'dinov2/configs/train/vit3d_highres.yaml' \\
      --output-dir '$OUTPUT_DIR' \\
      --pretrained-weights '$PRETRAINED_WEIGHTS' \\
      --dataset-name '$DATASET_NAME' \\
      --dataset-percent 100 \\
      --base-data-dir '$BASE_DATA_DIR' \\
      --segmentation-head 'ViTAdapterUNETR' \\
      --epochs 100 \\
      --epoch-length 300 \\
      --eval-iters 600 \\
      --warmup-iters 3000 \\
      --image-size 112 \\
      --batch-size 2 \\
      --num-workers 15 \\
      --learning-rate 1e-4 \\
      --cache-dir '$CACHE_DIR' \\
      --resize-scale 1.0 &&
    
    echo '=================================================' &&
    echo 'DINO training for fold {fold_number} completed!' &&
    echo 'Check output directory: $OUTPUT_DIR' &&
    echo '================================================='
    "
  scale: 1
  use_spot_instance: false
  schedule: null"""
    
    return yaml_content

def generate_test_yaml(fold_number, github_username="PeterMcGor", branch_name="fomo-finetuning-t2"):
    """Generate a test YAML with reduced parameters for quick validation"""
    
    yaml_content = f"""schema_version: 2024.04.01
type: job
spec:
  name: dino3d-fomo-fold{fold_number}-test-499
  owner: dgm-ms-brain-mri/pedro-maciasgordaliza
  description: 'Quick test of 3D-DINO finetuning setup for fold {fold_number}'
  image: nvidia-oci/saturncloud/saturn-python-pytorch:2024.08.01
  instance_type: 1xA100
  environment_variables:
    FOLD_NUMBER: "{fold_number}"
    PYTHONPATH: "/home/jovyan/workspace/3D-DINO"
    BASE_DATA_DIR: "/home/jovyan/workspace/3D-DINO/dino_datasets/fomo-task2_2channels/"
    OUTPUT_DIR: "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/test_fold-499_{fold_number}"
    CACHE_DIR: "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/cache_dir_test_fold-499_{fold_number}"
    PRETRAINED_WEIGHTS: "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/Dino3d_last-models/highres_teacher_checkpoint.pth"
    DATASET_NAME: "fomo-task2_2channels_fold_{fold_number}"
  working_directory: /home/jovyan/workspace/3D-DINO
  token_scope: job:{{self}}:dask:write
  git_repositories:
    - url: git@github.com:{github_username}/3D-DINO.git
      path: /home/jovyan/workspace/3D-DINO
      public: false
  secrets: []
  shared_folders:
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/fomo25
      name: fomo25
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset
      name: openmind-dataset
    - owner: dgm-ms-brain-mri/pedro-maciasgordaliza
      path: /home/jovyan/shared/pedro-maciasgordaliza/ms-data
      name: ms-data
  command: |
    bash -c "
    echo '=================================================' &&
    echo 'DINO Training TEST - Fold {fold_number}' &&
    echo '=================================================' &&
    
    echo 'Step 1: Setup git repository...' &&
    git fetch origin {branch_name} &&
    git checkout {branch_name} &&
    git reset --hard origin/{branch_name} &&
    
    echo 'Step 2: Create conda environment...' &&
    conda create -n dino3d python=3.9 -y &&
    
    echo 'Step 3: Initialize and activate environment...' &&
    conda init bash &&
    source ~/.bashrc &&
    conda activate dino3d &&
    
    echo 'Step 4: Install requirements...' &&
    pip install -r requirements.txt --quiet &&
    
    echo 'Step 5: Create output directories...' &&
    mkdir -p $OUTPUT_DIR &&
    mkdir -p $CACHE_DIR &&
    
    echo 'Step 6: Verify environment...' &&
    python -c 'import torch; print(\\\"PyTorch version:\\\", torch.__version__); print(\\\"CUDA available:\\\", torch.cuda.is_available())' &&
    
    echo 'Step 7: Starting DINO training TEST - Fold {fold_number} (1 epoch only)...' &&
    python dinov2/eval/segmentation3d.py \\
      --config-file 'dinov2/configs/train/vit3d_highres.yaml' \\
      --output-dir '$OUTPUT_DIR' \\
      --pretrained-weights '$PRETRAINED_WEIGHTS' \\
      --dataset-name '$DATASET_NAME' \\
      --dataset-percent 100 \\
      --base-data-dir '$BASE_DATA_DIR' \\
      --segmentation-head 'ViTAdapterUNETR' \\
      --epochs 1 \\
      --epoch-length 10 \\
      --eval-iters 5 \\
      --warmup-iters 5 \\
      --image-size 112 \\
      --batch-size 2 \\
      --num-workers 15 \\
      --learning-rate 1e-4 \\
      --cache-dir '$CACHE_DIR' \\
      --resize-scale 1.0 &&
    
    echo '=================================================' &&
    echo 'DINO training TEST for fold {fold_number} completed!' &&
    echo '================================================='
    "
  scale: 1
  use_spot_instance: false
  schedule: null"""
    
    return yaml_content

def main():
    """Main function to generate all YAML files"""
    
    parser = argparse.ArgumentParser(description="Generate YAML job files for cross-validation training")
    parser.add_argument("--github-username", default="PeterMcGor", 
                       help="GitHub username (default: PeterMcGor)")
    parser.add_argument("--branch-name", default="fomo-finetuning-t2", 
                       help="Branch name (default: fomo-finetuning-t2)")
    parser.add_argument("--total-folds", type=int, default=5, 
                       help="Total number of folds (default: 5)")
    parser.add_argument("--output-dir", default="saturn_utils/job_yamls", 
                       help="Output directory for YAML files (default: saturn_utils/job_yamls)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Generate test YAMLs with reduced parameters")
    parser.add_argument("--use-spot-instances", action="store_true",
                       help="Use spot instances for cost savings (default: false for reliability)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode = "test" if args.test_mode else "full training"
    spot_info = "with spot instances" if args.use_spot_instances else "with guaranteed instances"
    print(f"Generating YAML files for {args.total_folds} folds ({mode}, {spot_info})...")
    print(f"GitHub: {args.github_username}, Branch: {args.branch_name}")
    
    # Generate YAML for each fold
    for fold_num in range(args.total_folds):
        if args.test_mode:
            yaml_content = generate_test_yaml(fold_num, args.github_username, args.branch_name)
            filename = output_dir / f"fold{fold_num}_test.yaml"
        else:
            yaml_content = generate_fold_yaml(fold_num, args.total_folds, args.github_username, args.branch_name)
            filename = output_dir / f"fold{fold_num}_training.yaml"
        
        # Update spot instance setting if requested
        if args.use_spot_instances:
            yaml_content = yaml_content.replace("use_spot_instance: false", "use_spot_instance: true")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(yaml_content)
        
        print(f"Generated: {filename}")
    
    print(f"\nAll YAML files generated in '{output_dir}' directory")
    print("\nNext steps:")
    if args.test_mode:
        print("1. Test with: sc job create <test_yaml_file> --start")
        print("2. Once tests pass, generate full training YAMLs (without --test-mode)")
    else:
        print("1. Review the generated YAML files")
        print("2. Submit jobs using: sc job create <yaml_file> --start")
        print("3. Monitor jobs using: sc job list")
    
    if args.use_spot_instances:
        print("\nNote: Using spot instances may cause delays if resources aren't available.")
        print("Consider using guaranteed instances for time-sensitive training.")
    else:
        print("\nNote: Using guaranteed instances for reliable execution.")
        print("Add --use-spot-instances flag for cost savings (with potential delays).")

if __name__ == "__main__":
    main()