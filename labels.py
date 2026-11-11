import torch


def convert_msd_labels(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert an MSD Task01 label mask (values 0-3) into 3 binary channels.

    Input:  (1, H, W, D) with values 0=background, 1=edema,
            2=non-enhancing tumor, 3=enhancing tumor
    Output: (3, H, W, D) as float: [WT, TC, ET]
        WT (whole tumor)     = labels 1, 2, 3
        TC (tumor core)      = labels 2, 3
        ET (enhancing tumor) = label 3
    """
    wt = mask > 0
    tc = (mask == 2) | (mask == 3)
    et = mask == 3
    return torch.cat([wt, tc, et], dim=0).float()


if __name__ == "__main__":
    # Tiny test: one row of voxels with labels 0, 1, 2, 3
    dummy = torch.tensor([0, 1, 2, 3]).reshape(1, 1, 1, 4)
    out = convert_msd_labels(dummy)
    print(out.shape)  # expect (3, 1, 1, 4)
    print(out[0].flatten().tolist())  # WT: [0, 1, 1, 1]
    print(out[1].flatten().tolist())  # TC: [0, 0, 1, 1]
    print(out[2].flatten().tolist())  # ET: [0, 0, 0, 1]
