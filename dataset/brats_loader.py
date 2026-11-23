import os
import glob
import json
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandSpatialCropd,
    ConvertToMultiChannelBasedOnBratsClassesd,
)
from monai.data import NibabelReader
def get_msd_dataloaders(data_dir, batch_size=2, roi_size=(128, 128, 128)):
    """
    Loads MSD Task01_BrainTumour dataset and returns train and validation DataLoaders.
    """
    dataset_json_path = os.path.join(data_dir, "dataset.json")
    
    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)
    
    # Build list of dictionaries {'image': path, 'label': path}
    data_dicts = []
    for item in dataset_info["training"]:
        image_path = os.path.join(data_dir, item["image"].replace("./", ""))
        label_path = os.path.join(data_dir, item["label"].replace("./", ""))
        data_dicts.append({"image": image_path, "label": label_path})
    
    # Split 80% training, 20% validation
    split_idx = int(len(data_dicts) * 0.8)
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]
    
    # Medical image preprocessing transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NibabelReader()),
        EnsureChannelFirstd(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=NibabelReader()),
        EnsureChannelFirstd(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Path where Task01_BrainTumour is extracted
    sample_data_dir = "dataset/Task01_BrainTumour"
    if os.path.exists(sample_data_dir):
        train_loader, val_loader = get_msd_dataloaders(sample_data_dir)
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}") # Expected: (2, 4, 128, 128, 128)
        print(f"Batch label shape: {batch['label'].shape}") # Expected: (2, 3, 128, 128, 128)
    else:
        print(f"Waiting for dataset extraction in: {sample_data_dir}")
