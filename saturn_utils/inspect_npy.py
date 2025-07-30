import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_npy_pkl_data(data_path: str, file_prefix: str = "FOMO2_sub_", num_samples: int = 3):
    """
    Comprehensive inspection of .npy and .pkl files to understand data structure.
    
    Args:
        data_path: Path to your preprocessed data folder
        file_prefix: Prefix of your files
        num_samples: Number of samples to inspect
    """
    
    data_dir = Path(data_path)
    
    # Get available subjects
    npy_files = list(data_dir.glob(f"{file_prefix}*.npy"))
    subjects = []
    
    for npy_file in npy_files[:num_samples]:
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        pkl_file = data_dir / f"{file_prefix}{subject_id}.pkl"
        
        if pkl_file.exists():
            subjects.append(subject_id)
    
    print(f"Inspecting {len(subjects)} subjects: {subjects}")
    print("=" * 80)
    
    for i, subject_id in enumerate(subjects):
        print(f"\n📁 SUBJECT {subject_id} (Sample {i+1}/{len(subjects)})")
        print("-" * 50)
        
        npy_path = data_dir / f"{file_prefix}{subject_id}.npy"
        pkl_path = data_dir / f"{file_prefix}{subject_id}.pkl"
        
        # Inspect NPY file
        inspect_npy_file(npy_path)
        
        # Inspect PKL file  
        inspect_pkl_file(pkl_path)
        
        print("-" * 50)
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    provide_recommendations(data_dir, file_prefix, subjects[0] if subjects else None)

def inspect_npy_file(npy_path: Path):
    """Detailed inspection of .npy file"""
    print(f"🔍 NPY FILE: {npy_path.name}")
    
    try:
        # Load the array (with pickle support for object arrays)
        data = np.load(npy_path, allow_pickle=True)
        
        print(f"   💾 File size: {npy_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Analyze the loaded data structure
        if isinstance(data, dict):
            print(f"   🗂️  NPY contains dictionary with keys: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"      '{key}': array shape {value.shape}, dtype {value.dtype}")
                    if len(value.shape) >= 3:
                        print(f"         min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")
                else:
                    print(f"      '{key}': {type(value)} = {str(value)[:100]}...")
        
        elif isinstance(data, np.ndarray):
            # Regular numpy array
            print(f"   📊 Shape: {data.shape}")
            print(f"   🏷️  Data type: {data.dtype}")
            print(f"   📈 Min value: {data.min():.4f}")
            print(f"   📈 Max value: {data.max():.4f}")
            print(f"   📈 Mean value: {data.mean():.4f}")
            
            # Analyze dimensions
            analyze_array_dimensions(data)
        
        elif isinstance(data, (list, tuple)):
            print(f"   📋 NPY contains {type(data).__name__} of length: {len(data)}")
            for i, item in enumerate(data[:3]):  # Show first 3 items
                if isinstance(item, np.ndarray):
                    print(f"      Item {i}: array shape {item.shape}, dtype {item.dtype}")
                else:
                    print(f"      Item {i}: {type(item)}")
        
        else:
            print(f"   ❓ NPY contains: {type(data)}")
            print(f"   📄 Content preview: {str(data)[:200]}...")
        
    except Exception as e:
        print(f"   ❌ Error loading NPY: {e}")

def analyze_array_dimensions(data):
    """Analyze array dimensions and provide interpretation"""
    if len(data.shape) == 3:
        print(f"   📐 3D volume: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}")
        print("   💡 Interpretation: Single-channel 3D volume (D, H, W)")
    
    elif len(data.shape) == 4:
        print(f"   📐 4D data: {data.shape[0]} channels x {data.shape[1]} x {data.shape[2]} x {data.shape[3]}")
        print("   💡 Interpretation: Multi-channel 3D volume (C, D, H, W)")
        
        # Show info for each channel
        for ch in range(min(data.shape[0], 5)):  # Show up to 5 channels
            ch_data = data[ch]
            print(f"      Channel {ch}: min={ch_data.min():.3f}, max={ch_data.max():.3f}, mean={ch_data.mean():.3f}")
    
    elif len(data.shape) == 5:
        print(f"   📐 5D data: {data.shape}")
        print("   💡 Interpretation: Possibly batch dimension or time series")
    
    else:
        print(f"   ❓ Unusual shape: {data.shape}")
    
    # Check for common medical imaging patterns
    analyze_medical_imaging_patterns(data)

def inspect_pkl_file(pkl_path: Path):
    """Detailed inspection of .pkl file"""
    print(f"🔍 PKL FILE: {pkl_path.name}")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   🏷️  Data type: {type(data)}")
        print(f"   💾 File size: {pkl_path.stat().st_size / (1024*1024):.2f} MB")
        
        if isinstance(data, dict):
            print(f"   🗂️  Dictionary keys: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"      '{key}': array shape {value.shape}, dtype {value.dtype}")
                    if len(value.shape) <= 3:  # Show some stats for small arrays
                        print(f"         min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")
                elif isinstance(value, (str, Path)):
                    print(f"      '{key}': path/string = {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"      '{key}': {type(value).__name__} of length {len(value)}")
                    if len(value) > 0:
                        print(f"         first element type: {type(value[0])}")
                else:
                    print(f"      '{key}': {type(value)} = {str(value)[:100]}...")
            print('Original spacing',data['original_spacing'], 'New spacing',data['new_spacing'])
        
        elif isinstance(data, np.ndarray):
            print(f"   📊 Array shape: {data.shape}")
            print(f"   🏷️  Array dtype: {data.dtype}")
            print(f"   📈 Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
        
        elif isinstance(data, (list, tuple)):
            print(f"   📋 {type(data).__name__} of length: {len(data)}")
            if len(data) > 0:
                print(f"   🔍 First element type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"   📊 First element shape: {data[0].shape}")
        
        else:
            print(f"   📄 Content preview: {str(data)[:200]}...")
    
    except Exception as e:
        print(f"   ❌ Error loading PKL: {e}")

def analyze_medical_imaging_patterns(data):
    """Analyze if the data follows common medical imaging patterns"""
    
    # Check if it's likely medical imaging data
    if len(data.shape) >= 3:
        # Check intensity ranges (common in medical imaging)
        if data.min() >= 0 and data.max() <= 1:
            print("   🏥 Medical pattern: Normalized intensities [0,1]")
        elif data.min() >= 0 and data.max() <= 255:
            print("   🏥 Medical pattern: 8-bit intensities [0,255]")
        elif data.min() < 0 and abs(data.min()) > 100:
            print("   🏥 Medical pattern: Possible raw medical imaging (CT/MRI Hounsfield/intensity units)")
        
        # Check for typical medical imaging dimensions
        dims = data.shape[-3:] if len(data.shape) >= 3 else data.shape
        if all(d >= 64 and d <= 512 for d in dims):
            print("   🏥 Medical pattern: Typical medical imaging dimensions")

def provide_recommendations(data_dir: Path, file_prefix: str, sample_subject: str):
    """Provide recommendations based on the inspection"""
    
    if not sample_subject:
        print("❌ No valid samples found!")
        return
    
    print("🎯 RECOMMENDATIONS FOR 3D DINO:")
    
    # Load sample data to make recommendations
    npy_path = data_dir / f"{file_prefix}{sample_subject}.npy"
    pkl_path = data_dir / f"{file_prefix}{sample_subject}.pkl"
    
    try:
        npy_data = np.load(npy_path, allow_pickle=True)
        
        if isinstance(npy_data, dict):
            print("✅ NPY files contain dictionaries - likely preprocessed data + labels")
            if 'data' in npy_data and 'seg' in npy_data:
                print("   💡 Standard format with 'data' (image) and 'seg' (segmentation)")
                data_array = npy_data['data']
                seg_array = npy_data['seg']
                print(f"   📊 Image shape: {data_array.shape}")
                print(f"   📊 Label shape: {seg_array.shape}")
                print("   💡 Can extract these arrays for 3D DINO")
            else:
                print(f"   🔍 Dictionary keys: {list(npy_data.keys())}")
                print("   💡 Need to identify which keys contain image/label data")
        
        elif isinstance(npy_data, np.ndarray):
            if len(npy_data.shape) == 3:
                print("✅ NPY files are 3D single-channel - can use directly in 3D DINO")
                print("   💡 Use: 'image': 'path/to/file.npy' in JSON")
            
            elif len(npy_data.shape) == 4:
                num_channels = npy_data.shape[0]
                print(f"✅ NPY files are 4D with {num_channels} channels - can use directly in 3D DINO")
                print("   💡 MONAI will automatically handle multi-channel .npy files")
                print("   💡 Use: 'image': 'path/to/file.npy' in JSON")
                print(f"   📝 The dataloader will see this as a {num_channels}-channel 3D volume")
            
            else:
                print(f"⚠️  NPY files have unusual shape {npy_data.shape}")
                print("   💡 You might need preprocessing or custom loading")
        
        else:
            print(f"⚠️  NPY files contain {type(npy_data)} - need custom handling")
            print("   💡 Might need to extract arrays from the object structure")
    
    except Exception as e:
        print(f"❌ Could not analyze NPY files: {e}")
    
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        if isinstance(pkl_data, np.ndarray):
            print("💡 PKL files contain arrays - convert to .npy for 3D DINO compatibility")
            print("   🔧 Conversion code:")
            print(f"      np.save('label.npy', pickle.load(open('file.pkl', 'rb')))")
        
        elif isinstance(pkl_data, dict):
            print("💡 PKL files contain dictionaries - check if they have useful data")
            if 'label' in pkl_data or 'seg' in pkl_data or 'mask' in pkl_data:
                print("   ✅ Seems to contain label/segmentation data")
            if 'spacing' in pkl_data or 'original_spacing' in pkl_data or 'new_spacing' in pkl_data:
                print("   ✅ Contains spatial metadata - useful for 3D DINO JSON format")
    
    except Exception as e:
        print(f"❌ Could not analyze PKL files: {e}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run this updated inspection on your data")
    print("2. If NPY files contain dictionaries, extract 'data' and 'seg' arrays")  
    print("3. Use PKL metadata (original_spacing, new_size) for 3D DINO JSON format")
    print("4. Create converter function to extract arrays and create proper JSON")
    print("5. Test loading with MONAI using allow_pickle=True")
    
    print("\n🔧 QUICK TEST CODE:")
    print("```python")
    print("import numpy as np")
    print("print(f'Type: {type(data)}')")
    print("if isinstance(data, dict):")
    print("    print(f'Keys: {list(data.keys())}')")
    print("```")

# Example usage
if __name__ == "__main__":
    # Update this path to your data
    data_path = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2"
    
    inspect_npy_pkl_data(
        data_path=data_path,
        file_prefix="FOMO2_sub_",
        num_samples=3  # Inspect first 3 subjects
    )