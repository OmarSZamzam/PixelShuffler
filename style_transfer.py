import argparse

import lpips
import numpy as np
import torch
from monai.losses import GlobalMutualInformationLoss
from monai.networks.blocks import Warp
from monai.networks.nets import UNet
from PIL import Image
from tqdm import tqdm

from utils import (compute_style_loss, dvf_to_ddf, load_image, set_seed,
                   tensor_to_numpy)

# Set random seed
set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Main function to perform the style transfer
def style_transfer(content_img_path, style_img_path, loss_modes, size=512, epochs=1000):
    # Load and preprocess images
    content_image = load_image(content_img_path, size).to(device)
    style_image = load_image(style_img_path, size).to(device)

    reg = UNet(
        spatial_dims=2,
        in_channels=6,
        out_channels=2,
        channels=(32, 64, 128, 128),
        strides=(2, 2, 2),
        num_res_units=3,
    ).to(device)

    warp_layer = Warp("bilinear", "border").to(device)

    optimizer = torch.optim.Adam(reg.parameters(), 3e-3)
    loss_fn_MI = GlobalMutualInformationLoss()
    loss_fn_LPIPS = lpips.LPIPS(net="alex").to(device)

    # Weight decay factor for LPIPS and Style loss over iterations
    weight_decay = np.linspace(0, 1, epochs*10)

    progress_bar = tqdm(range(epochs))

    moving_image_rigid = style_image.clone()
    fixed_image = content_image

    for i in progress_bar:
        optimizer.zero_grad()

        input_batch = torch.cat((moving_image_rigid, fixed_image), dim=1)
        dvf = reg(input_batch)
        ddf = dvf_to_ddf(dvf)

        moved_image = warp_layer(moving_image_rigid, ddf)

        total_loss = 0
        if "MI" in loss_modes:
            imgloss_MI = loss_fn_MI(moved_image, fixed_image)
            total_loss += imgloss_MI
        if "LPIPS" in loss_modes:
            imgloss_LPIPS = loss_fn_LPIPS(moved_image, fixed_image).mean()
            total_loss += 10 * weight_decay[i] * imgloss_LPIPS
        if "Style" in loss_modes:
            style_loss = compute_style_loss(moved_image, style_image)
            total_loss += 0.01 * weight_decay[i] * style_loss

        total_loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Mutual information = {-imgloss_MI.item():.6f}")

        moving_image_rigid = moved_image.detach()

    # Save final output image
    output_image = tensor_to_numpy(moved_image)
    output_img = Image.fromarray((output_image * 255).astype(np.uint8))
    output_img.save("Stylized.jpg")
    print("Stylized image saved as 'Stylized.jpg'")


# Parse command-line arguments
def main():
    parser = argparse.ArgumentParser(
        description="Style Transfer with PyTorch and MONAI"
    )
    parser.add_argument(
        "--content", type=str, required=True, help="Path to the content image"
    )
    parser.add_argument(
        "--style", type=str, required=True, help="Path to the style image"
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=["MI", "LPIPS", "Style"],
        default=["MI", "LPIPS", "Style"],
        help="Loss functions to include (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to run (default: 1000)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Size of the images (default: 512)"
    )

    args = parser.parse_args()

    style_transfer(
        args.content, args.style, args.losses, size=args.size, epochs=args.epochs
    )


if __name__ == "__main__":
    main()
