from torchvision import transforms as func
import torch
from torch.utils.data import Dataset

import random
import os
import numpy as np


class style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def freeze_initial_layers(model, freeze_up_to_layer=3):
    # The ResNet50 features block is typically named 'layerX' in PyTorch
    layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']

    for name, child in model.named_children():
        if name in layer_names[:freeze_up_to_layer]:
            for param in child.parameters():
                param.requires_grad = False
            print(f'Layer {name} has been frozen.')
        else:
            print(f'Layer {name} is trainable.')



# class CustomTransform:
#     def __init__(self, output_size=(CFG.resolution, CFG.resolution), radius_factor=0.9):
#         self.output_size = output_size
#         self.radius_factor = radius_factor
#
#     def __call__(self, img):
#         # Assuming img is a PIL Image
#         # Normalize and preprocess as previously defined
#         img = func.resize(img, int(min(img.size) / self.radius_factor))
#         img_tensor = func.to_tensor(img)
#         mean, std = img_tensor.mean([1, 2]), img_tensor.std([1, 2])
#         img_normalized = func.normalize(img_tensor, mean.tolist(), std.tolist())
#         kernel_size = 15
#         padding = kernel_size // 2
#         avg_pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
#         local_avg = avg_pool(img_normalized.unsqueeze(0)).squeeze(0)
#         img_subtracted = img_normalized - local_avg
#         center_crop_size = int(min(img_subtracted.shape[1:]) * self.radius_factor)
#         img_cropped = func.center_crop(img_subtracted, [center_crop_size, center_crop_size])
#
#         # Apply augmentations
#         img_resized = func.resize(img_cropped, self.output_size)
#
#         return img_resized
#
#
# class ImageTrainDataset(Dataset):
#     def __init__(
#             self,
#             folder,
#             data,
#             transforms,
#     ):
#         self.folder = folder
#         self.data = data
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         d = self.data.loc[index]
#         image = Image.open(f"{self.folder}{d.image}.jpeg")
#         image = self.transforms(image)
#         label = d.level
#
#         return image, torch.tensor(label, dtype=torch.long)