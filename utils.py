import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.networks.blocks import DVF2DDF
from PIL import Image
from torchvision.models import vgg19

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = (
    vgg19(pretrained=True).features[:16].to(device).eval()
)  # Use first few layers for feature extraction

# Disable gradient calculations for VGG to save memory and computation
for param in vgg.parameters():
    param.requires_grad = False


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix for a given set of features.

    Args:
        features (torch.Tensor): Input features of shape (batch, channels, height, width).

    Returns:
        torch.Tensor: Normalized Gram matrix of shape (batch, channels, channels).
    """
    (b, c, h, w) = features.size()  # batch, channels, height, width
    features = features.view(b, c, h * w)  # Reshape to (batch, channels, height*width)
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix
    return gram / (c * h * w)  # Normalize


# Function to compute Gram Style Loss
def gram_style_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram Style Loss between two images.

    Args:
        img1 (torch.Tensor): The first input image tensor.
        img2 (torch.Tensor): The second input image tensor.

    Returns:
        torch.Tensor: Style loss computed as the mean squared error between Gram matrices.
    """
    vgg = vgg19(pretrained=True).features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    features1 = vgg(img1)
    features2 = vgg(img2)

    gram1 = gram_matrix(features1)
    gram2 = gram_matrix(features2)

    style_loss = F.mse_loss(gram1, gram2)
    return style_loss


dvf_to_ddf = DVF2DDF()


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch, channels, height, width).

    Returns:
        numpy.ndarray: Corresponding NumPy array of shape (height, width, channels).
    """
    return tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()


def load_image(path: str, size: int | tuple[int, int]) -> torch.Tensor:
    """
    Load and preprocess an image from a file.

    Args:
        path (str): Path to the image file.
        size (int or tuple): Desired size for resizing the image.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, channels, height, width).
    """
    return transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )(Image.open(path).convert("RGB")).unsqueeze(0)


def compute_style_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute the style loss between two images using mean and standard deviation of features.

    Args:
        img1 (torch.Tensor): The first input image tensor.
        img2 (torch.Tensor): The second input image tensor.

    Returns:
        torch.Tensor: Style loss computed as the sum of norms of differences in mean and standard deviation of features.
    """
    preprocess = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    features1 = vgg(preprocess(img1))
    features2 = vgg(preprocess(img2))

    mean1, std1 = torch.mean(features1, dim=(2, 3)), torch.std(features1, dim=(2, 3))
    mean2, std2 = torch.mean(features2, dim=(2, 3)), torch.std(features2, dim=(2, 3))

    style_loss = torch.norm(mean1 - mean2) + torch.norm(std1 - std2)
    return style_loss
