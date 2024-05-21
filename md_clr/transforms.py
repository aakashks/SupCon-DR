from torchvision.transforms import functional as func
import torchvision.transforms as v2
import torch
from .config import CFG


class CustomTransform:
    def __init__(self, output_size=(CFG.resolution, CFG.resolution), radius_factor=0.9):
        self.output_size = output_size
        self.radius_factor = radius_factor

    def __call__(self, img):
        # Assuming img is a PIL Image
        # Normalize and preprocess as previously defined
        img = func.resize(img, int(min(img.size) / self.radius_factor))
        img_tensor = func.to_tensor(img)
        mean, std = img_tensor.mean([1, 2]), img_tensor.std([1, 2])
        img_normalized = func.normalize(img_tensor, mean.tolist(), std.tolist())
        kernel_size = 15
        padding = kernel_size // 2
        avg_pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        local_avg = avg_pool(img_normalized.unsqueeze(0)).squeeze(0)
        img_subtracted = img_normalized - local_avg
        center_crop_size = int(min(img_subtracted.shape[1:]) * self.radius_factor)
        img_cropped = func.center_crop(img_subtracted, [center_crop_size, center_crop_size])

        # Apply augmentations
        img_resized = func.resize(img_cropped, self.output_size)

        return img_resized


train_transforms = v2.Compose([
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
    v2.RandomRotation(degrees=(0, 90)),
    CustomTransform(),
    # v2.RandomResizedCrop(CFG.resolution, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToDtype(torch.float32, scale=False),
])

sclr_train_transforms = v2.Compose([
    v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
    v2.RandomRotation(degrees=(0, 90)),
    # v2.RandomGrayscale(p=0.2),
    CustomTransform(),
    # v2.RandomResizedCrop(CFG.resolution, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomGrayscale(p=0.2),
    v2.ToDtype(torch.float32, scale=False),
])
    

val_transforms = v2.Compose([
    CustomTransform(),
    v2.ToDtype(torch.float32, scale=False),
])
