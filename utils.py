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


def gram_matrix(features):
    (b, c, h, w) = features.size()  # batch, channels, height, width
    features = features.view(b, c, h * w)  # Reshape to (batch, channels, height*width)
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix
    return gram / (c * h * w)  # Normalize


# Function to compute Gram Style Loss
def gram_style_loss(img1, img2):
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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_to_numpy(tensor):
    return tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()


def load_image(path, size):
    return transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )(Image.open(path).convert("RGB")).unsqueeze(0)


def compute_style_loss(img1, img2):
    preprocess = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    features1 = vgg(preprocess(img1))
    features2 = vgg(preprocess(img2))

    mean1, std1 = torch.mean(features1, dim=(2, 3)), torch.std(features1, dim=(2, 3))
    mean2, std2 = torch.mean(features2, dim=(2, 3)), torch.std(features2, dim=(2, 3))

    style_loss = torch.norm(mean1 - mean2) + torch.norm(std1 - std2)
    return style_loss
