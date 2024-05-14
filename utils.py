import torch
from torch.utils.data import Dataset

import random
import os
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn

from PIL import Image


class ImageTrainDataset(Dataset):
    def __init__(
            self,
            folder,
            data,
            transforms,
    ):
        self.folder = folder
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data.loc[index]
        image = Image.open(f"{self.folder}{d.image}.jpeg")
        image = self.transforms(image)
        label = d.level

        return image, torch.tensor(label, dtype=torch.long)


def get_embeddings(model, data_loader):
    model.eval()

    # remove the last layer (fc) of model to obtain embeddings
    model = nn.Sequential(*list(model.children())[:-2])

    features = []
    targets = []

    total_len = len(data_loader)
    tk0 = tqdm(enumerate(data_loader), total=total_len)
    with torch.no_grad():
        for step, (images, labels) in tk0:
            images = images.to(device)
            target = labels.to(device)

            embds = model(images)

            features.append(embds.detach().cpu())
            targets.append(target.detach().cpu())

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)

    # # store the embeddings for future use
    # torch.save(features, os.path.join(wandb.run.dir, f"embeddings.pth"))
    # torch.save(targets, os.path.join(wandb.run.dir, f"targets.pth"))

    return features, targets


def plot_tsne(embeddings, labels):
    # Apply t-SNE to the embeddings
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings.numpy())

    # Define the number of unique labels/classes
    num_classes = len(np.unique(labels.numpy()))
    # Create a custom color map with specific color transitions
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list("Custom", colors, N=num_classes)

    # Create a boundary norm with boundaries and colors
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), cmap.N)

    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.5)
    colorbar = plt.colorbar(scatter, ticks=np.arange(num_classes))
    colorbar.set_label('Severity Level')
    colorbar.set_ticklabels(np.arange(num_classes))  # Set discrete labels if needed
    plt.title('t-SNE of Image Embeddings with Discrete Severity Levels')
    plt.xlabel('t-SNE Axis 1')
    plt.ylabel('t-SNE Axis 2')
    fg = wandb.Image(fig)
    wandb.log({"t-SNE": fg})
    plt.savefig(os.path.join(wandb.run.dir, f"tsne.png"), dpi=300, bbox_inches='tight')



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