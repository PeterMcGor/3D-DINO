import json
import os
from pathlib import Path
from typing import Dict, List, Union

def create_dino_datasets_from_nnunet(
    nnunet_raw_path: str,
    nnunet_preprocessed_path: str,
    output_dir: str,
    dataset_name: str = "Dataset498_FOMO-Men_No-SK_FL-DWI",
    use_multichannel: bool = True,
    prefix: str = "dino_fold_"
):
    """
    Convert nnUNet dataset to 3D DINO format for cross-validation.
    
    Args:
        nnunet_raw_path: Path to nnUNet_raw folder
        nnunet_preprocessed_path: Path to nnUNet_preprocessed folder  
        output_dir: Directory to save the DINO JSON files
        dataset_name: Name of the dataset (e.g., "Dataset498_FOMO-Men_No-SK_FL-DWI")
        use_multichannel: If True, include all channels as image1, image2, image3, etc.
        prefix: Prefix for output JSON files
    """
    
    # Paths
    raw_dataset_path = Path(nnunet_raw_path) / dataset_name
    preprocessed_dataset_path = Path(nnunet_preprocessed_path) / dataset_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    splits_file = preprocessed_dataset_path / "splits_final.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"splits_final.json not found at {splits_file}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get test subjects (always the same across folds)
    images_ts_path = raw_dataset_path / "imagesTs"
    labels_ts_path = raw_dataset_path / "labelsTs"
    
    # Extract test subject IDs from filenames
    test_subjects = []
    if images_ts_path.exists():
        test_files = list(images_ts_path.glob("*_0000.nii.gz"))
        test_subjects = [f.name.split("_0000")[0] for f in test_files]
    
    print(f"Found {len(test_subjects)} test subjects: {test_subjects}")
    print(f"Creating datasets for {len(splits)} folds")
    
    # Process each fold
    for fold_idx in range(len(splits)):
        fold_data = splits[fold_idx]
        
        # Get training and validation subject IDs for this fold
        train_subjects = [case.replace(".nii.gz", "") for case in fold_data["train"]]
        val_subjects = [case.replace(".nii.gz", "") for case in fold_data["val"]]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
        
        # Create DINO format dataset
        dino_dataset = {
            "training": [],
            "validation": [],
            "test": []
        }
        
        # Add training data
        for subject in train_subjects:
            sample_data = get_multichannel_paths(raw_dataset_path, subject, "Tr", use_multichannel)
            if sample_data:
                dino_dataset["training"].append(sample_data)
        
        # Add validation data
        for subject in val_subjects:
            sample_data = get_multichannel_paths(raw_dataset_path, subject, "Tr", use_multichannel)
            if sample_data:
                dino_dataset["validation"].append(sample_data)
        
        # Add test data
        for subject in test_subjects:
            sample_data = get_multichannel_paths(raw_dataset_path, subject, "Ts", use_multichannel)
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

def get_multichannel_paths(
    raw_dataset_path: Path, 
    subject: str, 
    split: str, 
    use_multichannel: bool = True
) -> Dict[str, str]:
    """
    Get image and label paths for a subject in 3D DINO format.
    
    Args:
        raw_dataset_path: Path to raw dataset
        subject: Subject ID (e.g., "sub_1")
        split: "Tr" for training or "Ts" for test
        use_multichannel: If True, include all channels as separate keys
    
    Returns:
        Dictionary with image1, image2, image3, label keys (3D DINO format)
    """
    images_dir = raw_dataset_path / f"images{split}"
    labels_dir = raw_dataset_path / f"labels{split}"
    
    label_path = labels_dir / f"{subject}.nii.gz"
    
    # Check if label exists
    if not label_path.exists():
        print(f"Warning: Label not found: {label_path}")
        return None
    
    if use_multichannel:
        # Find all available channels
        channels = []
        channel_idx = 0
        while True:
            channel_str = f"{channel_idx:04d}"
            channel_path = images_dir / f"{subject}_{channel_str}.nii.gz"
            
            if channel_path.exists():
                channels.append(str(channel_path))
                channel_idx += 1
            else:
                break
        
        if not channels:
            print(f"Warning: No image channels found for {subject}")
            return None
        
        # Create 3D DINO format with image1, image2, image3, etc.
        sample_data = {}
        for i, channel_path in enumerate(channels):
            sample_data[f"image{i+1}"] = channel_path
        sample_data["label"] = str(label_path)
        
        return sample_data
    
    else:
        # Single channel mode (use first channel only)
        image_path = images_dir / f"{subject}_0000.nii.gz"
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return None
        
        return {
            "image": str(image_path),
            "label": str(label_path)
        }



# Example usage
if __name__ == "__main__":
    # Example paths based on your directory structure
    nnunet_raw = "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_raw"
    nnunet_preprocessed = "/home/jovyan/shared/pedro-maciasgordaliza/ms-data/nnunet_folders/nnUNet_preprocessed"
    dino_dataset_name = "fomo-task2_3channels"
    output_directory = f"./dino_datasets/{dino_dataset_name}"
    
    # Create multi-channel datasets (recommended - matches BraTS format)
    create_dino_datasets_from_nnunet(
        nnunet_raw_path=nnunet_raw,
        nnunet_preprocessed_path=nnunet_preprocessed,
        output_dir=output_directory,
        dataset_name="Dataset498_FOMO-Men_No-SK_FL-DWI",
        prefix=f"{dino_dataset_name}_fold_",
        use_multichannel=True  # Creates image1, image2, image3 format
    )
    
    # For single-channel (if needed)
    # create_dino_datasets_from_nnunet(
    #     nnunet_raw_path=nnunet_raw,
    #     nnunet_preprocessed_path=nnunet_preprocessed,
    #     output_dir=output_directory,
    #     dataset_name="Dataset498_FOMO-Men_No-SK_FL-DWI",
    #     use_multichannel=False  # Uses only first channel
    # )