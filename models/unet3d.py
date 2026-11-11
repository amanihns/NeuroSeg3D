import torch
import torch.nn as nn
from monai.networks.nets import UNet

def get_neuroseg_model(in_channels=4, out_channels=3):
    """
    3D U-Net Architecture for NeuroSeg3D.
    
    in_channels=4: MRI sequences (T1, T1ce, T2, FLAIR)
    out_channels=3: Tumor regions (WT, TC, ET)
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    return model

if __name__ == "__main__":
    # Test model shape with dummy 3D input tensor
    dummy_input = torch.randn(1, 4, 128, 128, 128)
    model = get_neuroseg_model()
    output = model(dummy_input)
    print(f"Model initialized successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")