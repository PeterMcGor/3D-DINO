import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

def create_dino_datasets_from_npy_pkl(
    preprocessed_data_path: str,
    nnunet_preprocessed_path: str,
    output_dir: str,
    extracted_data_dir: str,
    experiment_name: str,
    dataset_name: str = "Dataset499_FOMO-Men_No-SK_FL-DWI",
    file_prefix: str = "FOMO2_sub_",
    num_image_channels: int = 3
):
    """
    Convert NPY/PKL dataset to 3D DINO format for cross-validation using nnUNet splits.
    
    Your NPY files have 4 channels:
    - Channel 0: DWI → extracted as {subject}_dwi.npy
    - Channel 1: T2FLAIR → extracted as {subject}_flair.npy
    - Channel 2: SWI_OR_T2STAR → extracted as {subject}_swi.npy
    - Channel 3: Segmentation Label → extracted as {subject}_label.npy
    
    Args:
        preprocessed_data_path: Path to folder containing .npy/.pkl files
        nnunet_preprocessed_path: Path to nnUNet_preprocessed folder (for splits)
        output_dir: Directory to save the experiment folder and JSON files
        extracted_data_dir: Directory to save extracted modality .npy files
        experiment_name: Name of experiment (used for folder name and JSON prefix)
        dataset_name: Name of the nnUNet dataset for getting splits
        file_prefix: Prefix of your .npy/.pkl files (e.g., "FOMO2_sub_")
        num_image_channels: Number of image channels to use in JSON (2 or 3)
                           2 = image1 (DWI) + image2 (T2FLAIR) + label
                           3 = image1 (DWI) + image2 (T2FLAIR) + image3 (SWI) + label
    """
    
    # Paths
    data_path = Path(preprocessed_data_path)
    preprocessed_dataset_path = Path(nnunet_preprocessed_path) / dataset_name
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create extracted data directory (separate location)
    extracted_path = Path(extracted_data_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)
    
    # Load splits from nnUNet
    splits_file = preprocessed_dataset_path / "splits_final.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"splits_final.json not found at {splits_file}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get all available subjects from the preprocessed data
    npy_files = list(data_path.glob(f"{file_prefix}*.npy"))
    available_subjects = []
    
    for npy_file in npy_files:
        # Extract subject ID from filename
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        pkl_file = data_path / f"{file_prefix}{subject_id}.pkl"
        
        if pkl_file.exists():
            available_subjects.append(subject_id)
        else:
            print(f"Warning: Missing PKL file for {subject_id}")
    
    print(f"Found {len(available_subjects)} subjects with both .npy and .pkl files")
    print(f"Available subjects: {sorted(available_subjects)}")
    print(f"Creating {num_image_channels}-channel experiment with {len(splits)} folds")
    
    # Extract individual modality channels
    print(f"\n🔄 Extracting individual modality channels...")
    
    # Check if extraction is needed
    sample_extracted = extracted_path / f"{file_prefix}{available_subjects[0]}_dwi.npy" if available_subjects else None
    if sample_extracted and sample_extracted.exists():
        print("  ✅ Extracted channels already exist, skipping extraction...")
    else:
        extract_modality_channels_from_npy(data_path, extracted_path, file_prefix, available_subjects)
    
    # Get test subjects from nnUNet structure (same for all folds)
    test_subjects = get_test_subjects_from_nnunet(nnunet_preprocessed_path, dataset_name, available_subjects)
    
    # Process each fold
    for fold_idx in range(len(splits)):
        fold_data = splits[fold_idx]
        
        # Get training and validation subject IDs for this fold
        train_subjects = [extract_subject_id(case) for case in fold_data["train"]]
        val_subjects = [extract_subject_id(case) for case in fold_data["val"]]
        
        # Filter to only include subjects that have preprocessed data
        train_subjects = [s for s in train_subjects if s in available_subjects]
        val_subjects = [s for s in val_subjects if s in available_subjects]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train subjects: {len(train_subjects)} - {train_subjects}")
        print(f"  Val subjects: {len(val_subjects)} - {val_subjects}")
        print(f"  Test subjects: {len(test_subjects)} - {test_subjects}")
        
        # Create DINO format dataset
        dino_dataset = {
            "training": [],
            "validation": [],
            "test": []
        }
        
        # Add training data
        for subject in train_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["training"].append(sample_data)
        
        # Add validation data
        for subject in val_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["validation"].append(sample_data)
        
        # Add test data
        for subject in test_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["test"].append(sample_data)
        
        # Save fold dataset
        output_file = output_path / f"{experiment_name}_fold_{fold_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(dino_dataset, f, indent=2)
        
        print(f"  Saved to: {output_file}")
        print(f"  Training samples: {len(dino_dataset['training'])}")
        print(f"  Validation samples: {len(dino_dataset['validation'])}")
        print(f"  Test samples: {len(dino_dataset['test'])}")

def get_test_subjects_from_nnunet(nnunet_preprocessed_path: str, dataset_name: str, available_subjects: List[str]) -> List[str]:
    """
    Get test subjects from nnUNet structure that also have preprocessed data.
    
    Args:
        nnunet_preprocessed_path: Path to nnUNet_preprocessed folder
        dataset_name: nnUNet dataset name
        available_subjects: List of subjects that have preprocessed .npy/.pkl files
    
    Returns:
        List of test subject IDs that have preprocessed data
    """
    # Try to find test subjects from nnUNet structure
    # This could be from nnUNet_raw/Dataset/imagesTs or from splits file
    nnunet_raw_path = Path(nnunet_preprocessed_path).parent / "nnUNet_raw" / dataset_name
    images_ts_path = nnunet_raw_path / "imagesTs"
    
    test_subjects = []
    
    if images_ts_path.exists():
        # Extract test subject IDs from nnUNet test files
        test_files = list(images_ts_path.glob("*_0000.nii.gz"))
        for test_file in test_files:
            subject_id = test_file.name.split("_0000")[0]
            # Convert nnUNet naming to our preprocessed naming
            extracted_id = extract_subject_id(subject_id + ".nii.gz")
            if extracted_id in available_subjects:
                test_subjects.append(extracted_id)
        
        print(f"Found {len(test_subjects)} test subjects from nnUNet structure: {test_subjects}")
    else:
        print(f"Warning: No test data found at {images_ts_path}")
        print("Using a subset of available subjects as test data")
        # Fallback: use some available subjects as test data
        test_subjects = available_subjects[-3:] if len(available_subjects) > 3 else []
    
    return test_subjects

def extract_modality_channels_from_npy(data_path: Path, extracted_path: Path, file_prefix: str, subjects: List[str]):
    """
    Extract each channel as a separate .npy file with modality names.
    
    Your data structure: 4 channels [DWI, T2FLAIR, SWI_OR_T2STAR, Label]
    
    Creates:
    - FOMO2_sub_X_dwi.npy (DWI - Channel 0)
    - FOMO2_sub_X_flair.npy (T2FLAIR - Channel 1) 
    - FOMO2_sub_X_swi.npy (SWI_OR_T2STAR - Channel 2)
    - FOMO2_sub_X_label.npy (Segmentation - Channel 3)
    """
    
    modality_info = {
        0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "swi", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging)"},
        3: {"name": "label", "description": "Segmentation Label"}
    }
    
    print("  Extracting modality channels:")
    for i, info in modality_info.items():
        print(f"    Channel {i}: {info['name']}.npy ({info['description']})")
    
    for subject_id in subjects:
        npy_path = data_path / f"{file_prefix}{subject_id}.npy"
        
        try:
            # Load the 4-channel data
            combined_data = np.load(npy_path, allow_pickle=True)
            
            if len(combined_data) != 4:
                print(f"Warning: Expected 4 channels, got {len(combined_data)} for {subject_id}")
                continue
            
            # Extract each channel with modality name
            for channel_idx, data_array in enumerate(combined_data):
                modality_name = modality_info[channel_idx]["name"]
                output_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
                
                np.save(output_path, data_array)
                print(f"  Extracted {modality_name}: {output_path.name} - Shape: {data_array.shape}")
                
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")

def get_modality_channel_paths(extracted_path: Path, file_prefix: str, subject_id: str, data_path: Path, num_image_channels: int = 3) -> Optional[Dict[str, str]]:
    """
    Get paths to individual modality files in simple 3D DINO format.
    Returns clean format without spacing/shape metadata.
    
    Args:
        num_image_channels: Number of image channels to include (2 or 3)
                           2 = image1 (dwi) + image2 (flair) + label
                           3 = image1 (dwi) + image2 (flair) + image3 (swi) + label
    """
    # Build the sample data dictionary (clean format)
    sample_data = {}
    
    # Define modality mapping to 3D DINO image keys
    modality_mapping = {
        1: "dwi",      # image1 = DWI
        2: "flair",    # image2 = T2FLAIR  
        3: "swi"       # image3 = SWI_OR_T2STAR
    }
    
    # Add image channels based on num_image_channels
    for i in range(1, num_image_channels + 1):
        modality_name = modality_mapping[i]
        image_key = f"image{i}"
        image_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
        
        if not image_path.exists():
            print(f"Warning: {modality_name} file not found: {image_path}")
            return None
        
        sample_data[image_key] = str(image_path)
    
    # Add label
    label_path = extracted_path / f"{file_prefix}{subject_id}_label.npy"
    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")
        return None
    
    sample_data["label"] = str(label_path)
    
    return sample_data

def get_metadata_from_pkl(pkl_path: Path) -> tuple:
    """
    Extract spacing and shape information from PKL metadata file.
    """
    try:
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Get spacing - prefer new_spacing over original_spacing
        if 'new_spacing' in metadata:
            spacing = metadata['new_spacing']
        elif 'original_spacing' in metadata:
            spacing = metadata['original_spacing']
        else:
            spacing = [1.0, 1.0, 1.0]  # Default spacing
            print(f"Warning: No spacing found in {pkl_path.name}, using default [1.0, 1.0, 1.0]")
        
        # Get shape - prefer new_size over size_after_transpose
        if 'new_size' in metadata:
            shape = metadata['new_size']
        elif 'size_after_transpose' in metadata:
            shape = metadata['size_after_transpose']
        else:
            # Try to infer from other metadata
            shape = [128, 128, 64]  # Default shape
            print(f"Warning: No shape found in {pkl_path.name}, using default [128, 128, 64]")
        
        return spacing, shape
        
    except Exception as e:
        print(f"Warning: Could not read metadata from {pkl_path}: {e}")
        return [1.0, 1.0, 1.0], [128, 128, 64]

def extract_subject_id(nnunet_case_name: str) -> str:
    """
    Extract subject ID from nnUNet case name.
    
    Args:
        nnunet_case_name: e.g., "sub_1.nii.gz" or "FOMO2_sub_1.nii.gz"
    
    Returns:
        Subject ID: e.g., "1"
    """
    # Remove .nii.gz extension
    case_name = nnunet_case_name.replace(".nii.gz", "")
    
    # Extract the number part
    if "sub_" in case_name:
        return case_name.split("sub_")[-1]
    else:
        return case_name

def create_both_experiments(
    preprocessed_data_path: str,
    nnunet_preprocessed_path: str,
    base_output_dir: str,
    extracted_data_dir: str,
    experiment1_name: str = "2channel_dwi_flair",
    experiment2_name: str = "3channel_dwi_flair_swi",
    dataset_name: str = "Dataset499_FOMO-Men_No-SK_FL-DWI",
    file_prefix: str = "FOMO2_sub_"
):
    """
    Create both 2-channel and 3-channel experiments with 5 folds each.
    Uses the same extracted modality channels for both experiments.
    
    Args:
        preprocessed_data_path: Path to folder containing .npy/.pkl files
        nnunet_preprocessed_path: Path to nnUNet_preprocessed folder (for splits)
        base_output_dir: Base directory for both experiments
        extracted_data_dir: Directory to save extracted modality .npy files
        experiment1_name: Name for 2-channel experiment (folder + JSON prefix)
        experiment2_name: Name for 3-channel experiment (folder + JSON prefix)
        dataset_name: Name of the nnUNet dataset for getting splits
        file_prefix: Prefix of your .npy/.pkl files
    """
    
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Creating BOTH experiments with shared modality extraction:")
    print(f"  📊 Experiment 1: {experiment1_name} (image1=DWI + image2=T2FLAIR + label)")
    print(f"  📊 Experiment 2: {experiment2_name} (image1=DWI + image2=T2FLAIR + image3=SWI + label)")
    print(f"  💾 Extracted data will be saved to: {extracted_data_dir}")
    print("=" * 80)
    
    # Extract modalities once (shared for both experiments)
    print("\n🔄 Extracting modality channels (shared for both experiments)...")
    print("-" * 50)
    
    shared_extracted_path = Path(extracted_data_dir)
    shared_extracted_path.mkdir(parents=True, exist_ok=True)
    
    # Get available subjects
    data_path = Path(preprocessed_data_path)
    npy_files = list(data_path.glob(f"{file_prefix}*.npy"))
    available_subjects = []
    
    for npy_file in npy_files:
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        pkl_file = data_path / f"{file_prefix}{subject_id}.pkl"
        if pkl_file.exists():
            available_subjects.append(subject_id)
    
    # Extract modality channels
    sample_extracted = shared_extracted_path / f"{file_prefix}{available_subjects[0]}_dwi.npy" if available_subjects else None
    if sample_extracted and sample_extracted.exists():
        print("  ✅ Extracted channels already exist, skipping extraction...")
    else:
        extract_modality_channels_from_npy(data_path, shared_extracted_path, file_prefix, available_subjects)
    
    # Experiment 1: 2-channel (image1=dwi + image2=flair)
    print(f"\n🧪 EXPERIMENT 1: {experiment1_name}")
    print("-" * 50)
    
    # NOTE: Extraction is skipped for individual calls if using create_both_experiments()
    # because the channels are already extracted to the shared extracted_data_dir
    create_dino_datasets_from_npy_pkl(
        preprocessed_data_path=preprocessed_data_path,
        nnunet_preprocessed_path=nnunet_preprocessed_path,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=experiment1_name,
        dataset_name=dataset_name,
        file_prefix=file_prefix,
        num_image_channels=2
    )
    
    # Experiment 2: 3-channel (image1=dwi + image2=flair + image3=swi)
    print(f"\n🧪 EXPERIMENT 2: {experiment2_name}")
    print("-" * 50)
    
    # NOTE: Extraction is skipped for individual calls if using create_both_experiments()
    # because the channels are already extracted to the shared extracted_data_dir
    create_dino_datasets_from_npy_pkl(
        preprocessed_data_path=preprocessed_data_path,
        nnunet_preprocessed_path=nnunet_preprocessed_path,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=experiment2_name,
        dataset_name=dataset_name,
        file_prefix=file_prefix,
        num_image_channels=3
    )
    
    print("\n" + "=" * 80)
    print("✅ BOTH EXPERIMENTS COMPLETED!")
    print("📁 Output structure:")
    print(f"   {extracted_data_dir}/                     # Extracted modality files")
    print("   ├── FOMO2_sub_X_dwi.npy              # DWI")
    print("   ├── FOMO2_sub_X_flair.npy            # T2FLAIR")
    print("   ├── FOMO2_sub_X_swi.npy              # SWI_OR_T2STAR")
    print("   └── FOMO2_sub_X_label.npy            # Segmentation")
    print(f"   {base_output_dir}/")
    print(f"   ├── {experiment1_name}/")
    print(f"   │   ├── {experiment1_name}_fold_0.json    # 2-channel: image1=dwi, image2=flair, label + test")
    print(f"   │   ├── {experiment1_name}_fold_1.json")
    print("   │   ├── ... (5 folds total)")
    print(f"   │   └── {experiment1_name}_fold_4.json")
    print(f"   └── {experiment2_name}/")
    print(f"       ├── {experiment2_name}_fold_0.json    # 3-channel: image1=dwi, image2=flair, image3=swi, label + test")
    print(f"       ├── {experiment2_name}_fold_1.json")
    print("       ├── ... (5 folds total)")
    print(f"       └── {experiment2_name}_fold_4.json")
    
    print("\n📋 JSON format (simple, clean):")
    print('   {')
    print('     "training": [{"image1": "path/dwi.npy", "image2": "path/flair.npy", "label": "path/label.npy"}],')
    print('     "validation": [{"image1": "path/dwi.npy", "image2": "path/flair.npy", "label": "path/label.npy"}],')
    print('     "test": [{"image1": "path/dwi.npy", "image2": "path/flair.npy", "label": "path/label.npy"}]')
    print('   }')

def process_folds_for_experiment(
    preprocessed_data_path: str,
    nnunet_preprocessed_path: str,
    output_dir: str,
    extracted_path: Path,
    dataset_name: str,
    file_prefix: str,
    prefix: str,
    num_image_channels: int
):
    """
    Process folds for a specific experiment without re-extracting channels.
    """
    # Paths
    data_path = Path(preprocessed_data_path)
    preprocessed_dataset_path = Path(nnunet_preprocessed_path) / dataset_name
    output_path = Path(output_dir)
    
    # Load splits from nnUNet
    splits_file = preprocessed_dataset_path / "splits_final.json"
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get available subjects
    npy_files = list(data_path.glob(f"{file_prefix}*.npy"))
    available_subjects = []
    
    for npy_file in npy_files:
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        pkl_file = data_path / f"{file_prefix}{subject_id}.pkl"
        if pkl_file.exists():
            available_subjects.append(subject_id)
    
    # Get test subjects from nnUNet structure
    test_subjects = get_test_subjects_from_nnunet(nnunet_preprocessed_path, dataset_name, available_subjects)
    
    # Process each fold
    for fold_idx in range(len(splits)):
        fold_data = splits[fold_idx]
        
        # Get training and validation subject IDs for this fold
        train_subjects = [extract_subject_id(case) for case in fold_data["train"]]
        val_subjects = [extract_subject_id(case) for case in fold_data["val"]]
        
        # Filter to only include subjects that have preprocessed data
        train_subjects = [s for s in train_subjects if s in available_subjects]
        val_subjects = [s for s in val_subjects if s in available_subjects]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train subjects: {len(train_subjects)} - {train_subjects}")
        print(f"  Val subjects: {len(val_subjects)} - {val_subjects}")
        print(f"  Test subjects: {len(test_subjects)} - {test_subjects}")
        
        # Create DINO format dataset
        dino_dataset = {
            "training": [],
            "validation": [],
            "test": []
        }
        
        # Add training data
        for subject in train_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["training"].append(sample_data)
        
        # Add validation data
        for subject in val_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["validation"].append(sample_data)
        
        # Add test data
        for subject in test_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels)
            if sample_data:
                dino_dataset["test"].append(sample_data)
        
        # Save fold dataset
        output_file = output_path / f"{prefix}{fold_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(dino_dataset, f, indent=2)
        
        print(f"  Saved to: {output_file}")
        print(f"  Training samples: {len(dino_dataset['training'])}")
        print(f"  Validation samples: {len(dino_dataset['validation'])}")
        print(f"  Test samples: {len(dino_dataset['test'])}")

# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    preprocessed_data = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2/"
    nnunet_preprocessed = "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed"
    output_directory = "./dino_datasets/fomo_experiments"
    extracted_directory = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2_extracted_modalities"  # Separate location for extracted files
    
    # Create both experiments at once (RECOMMENDED)
    create_both_experiments(
        preprocessed_data_path=preprocessed_data,
        nnunet_preprocessed_path=nnunet_preprocessed,
        base_output_dir=output_directory,
        extracted_data_dir=extracted_directory,
        experiment1_name="fomo-task2_2channels_mimic",           # Custom name for 2-channel experiment
        experiment2_name="fomo-task2_3channels_mimic",       # Custom name for 3-channel experiment
        dataset_name="Dataset499_FOMO-Men_No-SK_FL-DWI",
        file_prefix="FOMO2_sub_"
    )
    
    # Or create individual experiments:
    
    # # Single 2-channel experiment
    # create_dino_datasets_from_npy_pkl(
    #     preprocessed_data_path=preprocessed_data,
    #     nnunet_preprocessed_path=nnunet_preprocessed,
    #     output_dir=output_directory,
    #     extracted_data_dir=extracted_directory,
    #     experiment_name="my_2channel_exp",         # Custom experiment name
    #     dataset_name="Dataset499_FOMO-Men_No-SK_FL-DWI",
    #     file_prefix="FOMO2_sub_",
    #     num_image_channels=2  # image1=dwi + image2=flair + label
    # )
    
    # # Single 3-channel experiment  
    # create_dino_datasets_from_npy_pkl(
    #     preprocessed_data_path=preprocessed_data,
    #     nnunet_preprocessed_path=nnunet_preprocessed,
    #     output_dir=output_directory,
    #     extracted_data_dir=extracted_directory,
    #     experiment_name="my_3channel_exp",         # Custom experiment name
    #     dataset_name="Dataset499_FOMO-Men_No-SK_FL-DWI",
    #     file_prefix="FOMO2_sub_",
    #     num_image_channels=3  # image1=dwi + image2=flair + image3=swi + label
    # )