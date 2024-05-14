OUTPUT_FOLDER = "/scratch/aakash_ks.iitr/dr-scnn/"
DATA_FOLDER = "/scratch/aakash_ks.iitr/data/diabetic-retinopathy/"
# TRAIN_DATA_FOLDER = DATA_FOLDER + 'resized_train/'
TRAIN_DATA_FOLDER = DATA_FOLDER + 'resized_train_c/'

TEST_DATA_FOLDER = DATA_FOLDER + 'test/'

import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 100

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms import v2

import timm

NUM_CLASSES = 5


class CFG:
    seed = 42
    N_folds = 6
    train_folds = [0, ]  # [0,1,2,3,4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    apex = True  # use half precision
    workers = 16

    model_name = "resnet50.a1_in1k"
    epochs = 10
    cropped = True
    # weights =  torch.tensor([0.206119, 0.793881],dtype=torch.float32)

    clip_val = 1000.
    batch_size = 64
    # gradient_accumulation_steps = 1

    lr = 1e-3
    weight_decay = 1e-2

    resolution = 224
    samples_per_class = 1000
    frozen_layers = 0


import wandb

run = wandb.init(
    project="hello-world",
    dir=OUTPUT_FOLDER,
    config={
        k: v for k, v in CFG.__dict__.items() if not k.startswith('__')}
)

device = torch.device(CFG.device)

# train_data = pd.read_csv(os.path.join(DATA_FOLDER, 'trainLabels.csv'))
train_data = pd.read_csv(os.path.join(DATA_FOLDER, 'trainLabels_cropped.csv')).sample(frac=1).reset_index(drop=True)

# remove all images from the csv if they are not in the folder
lst = map(lambda x: x[:-5], os.listdir(TRAIN_DATA_FOLDER))
train_data = train_data[train_data.image.isin(lst)]

# take only 100 samples from each class
train_data = train_data.groupby('level').head(CFG.samples_per_class).reset_index(drop=True)

from torchvision.transforms import functional as func
from utils import *


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


# train_transforms = CustomTransform()

train_transforms = v2.Compose([
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
    v2.RandomRotation(degrees=(0, 90)),
    CustomTransform(),
    # v2.RandomResizedCrop(CFG.resolution, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToDtype(torch.float32, scale=False),
])

val_transforms = v2.Compose([
    CustomTransform(),
    v2.ToDtype(torch.float32, scale=False),
])

# # visualize the transformations
# train_dataset = ImageTrainDataset(TRAIN_DATA_FOLDER, train_data, train_transforms)
# image, label = train_dataset[11]
# transformed_img_pil = func.to_pil_image(image)
# plt.imshow(transformed_img_pil)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score


def evaluate_model(cfg, feature_extractor, classifier, data_loader, loss_criterion, epoch=-1):
    loss_fn = loss_criterion

    model = classifier
    model.eval()

    val_loss = 0

    targets = []
    predictions = []

    total_len = len(data_loader)
    tk0 = tqdm(enumerate(data_loader), total=total_len)

    with torch.no_grad():
        for step, (images, labels) in tk0:
            images = images.to(device)
            target = labels.to(device)

            with torch.no_grad():
                features = feature_extractor(images)

            logits = model(features)

            loss = loss_fn(logits, target)
            val_loss += loss.item()

            targets.append(target.detach().cpu())
            predictions.append(logits.detach().cpu())
            del images, target, logits

    targets = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions, dim=0)
    probabilities = F.softmax(predictions, dim=1)

    val_loss /= total_len

    predicted_classes = predictions.argmax(dim=1)

    try:
        wandb.log({"roc": wandb.plot.roc_curve(targets.numpy(), probabilities.numpy())})
        roc_auc = roc_auc_score(targets.numpy(), probabilities.numpy(), multi_class='ovo')

    except:
        roc_auc = 0

    # Calculate accuracy
    accuracy = accuracy_score(targets.numpy(), predicted_classes.numpy())

    precision = precision_score(targets.numpy(), predicted_classes.numpy(), average='weighted')

    print(
        f'Epoch {epoch}: validation loss = {val_loss:.4f} auc = {roc_auc:.4f} accuracy = {accuracy:.4f} precision = {precision:.4f}')
    return val_loss, roc_auc, accuracy, precision


def train_epoch(cfg, feature_extractor, classifier, train_loader, loss_criterion, optimizer, scheduler, epoch):
    loss_fn = loss_criterion

    model = classifier
    model.train()

    train_loss = 0
    learning_rate_history = []

    targets = []
    predictions = []

    total_len = len(train_loader)
    tk0 = tqdm(enumerate(train_loader), total=total_len)
    for step, (images, labels) in tk0:
        images = images.to(device, non_blocking=True)
        target = labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = feature_extractor(images)

        logits = model(features)
        loss = loss_fn(logits, target)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_val)

        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is None:
            lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        tk0.set_description(
            f"Epoch {epoch} training {step + 1}/{total_len} [LR {lr:0.6f}] - loss: {train_loss / (step + 1):.4f}")

        learning_rate_history.append(lr)

        targets.append(target.detach().cpu())
        predictions.append(logits.detach().cpu())
        del images, target

    targets = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions, dim=0)
    probabilities = F.softmax(predictions, dim=1)

    train_loss /= total_len
    predicted_classes = predictions.argmax(dim=1)

    try:
        roc_auc = roc_auc_score(targets.numpy(), probabilities.numpy(), multi_class='ovo')
    except ValueError:
        roc_auc = 0

    # Calculate accuracy
    accuracy = accuracy_score(targets.numpy(), predicted_classes.numpy())

    precision = precision_score(targets.numpy(), predicted_classes.numpy(), average='weighted')

    print(
        f'Epoch {epoch}: training loss = {train_loss:.4f} auc = {roc_auc:.4f} accuracy = {accuracy:.4f} precision = {precision:.4f}')
    return train_loss, learning_rate_history, roc_auc, accuracy, precision


from sklearn.model_selection import StratifiedKFold

sgkf = StratifiedKFold(n_splits=CFG.N_folds, random_state=CFG.seed, shuffle=True)
for i, (train_index, test_index) in enumerate(sgkf.split(train_data["image"].values, train_data["level"].values)):
    train_data.loc[test_index, "fold"] = i


class LinearClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def create_model():
    # get the feature extractor
    feature_extractor = timm.create_model(CFG.model_name, num_classes=NUM_CLASSES, pretrained=False)
    feature_extractor.load_state_dict(torch.load(OUTPUT_FOLDER + 'tl_model.pth'))
    
    # remove the final layer
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])

    # create a simple linear classifier
    classifier = LinearClassifier()
    return feature_extractor.to(device), classifier.to(device)


for FOLD in CFG.train_folds:
    seed_everything(CFG.seed)

    # PREPARE DATA
    fold_train_data = train_data[train_data["fold"] != FOLD].reset_index(drop=True)
    fold_valid_data = train_data[train_data["fold"] == FOLD].reset_index(drop=True)

    train_dataset = ImageTrainDataset(TRAIN_DATA_FOLDER, fold_train_data, transforms=train_transforms)
    valid_dataset = ImageTrainDataset(TRAIN_DATA_FOLDER, fold_valid_data, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.workers,
        pin_memory=True,
        drop_last=False,
    )

    # PREPARE MODEL, OPTIMIZER AND SCHEDULER
    feature_extractor, model = create_model()
    feature_extractor.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6,
                                                           T_max=CFG.epochs * len(train_loader))

    loss_criterion = nn.CrossEntropyLoss()

    # TRAIN FOLD
    best_score = 0

    wandb.run.tags = [f"fold_{FOLD}"]

    for epoch in range(0, CFG.epochs):
        train_loss, train_lr, train_auc, train_accuracy, train_precision = train_epoch(CFG, feature_extractor,
                                                                                       model, train_loader,
                                                                                       loss_criterion, optimizer,
                                                                                       scheduler, epoch)

        val_loss, val_auc, val_accuracy, val_precision = evaluate_model(CFG, feature_extractor, model, valid_loader,
                                                                        loss_criterion, epoch)

        # Log metrics to wandb
        wandb.log({
            'train_loss': train_loss,
            'train_auc': train_auc,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'learning_rate': train_lr[-1]  # Log the last learning rate of the epoch
        })

        if (val_accuracy > best_score):
            print(f"{style.GREEN}New best score: {best_score:.4f} -> {val_accuracy:.4f}{style.END}")
            best_score = val_accuracy
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'best_lc_fold_{FOLD}.pth'))

    # plot a tsne plot of all the images using embeddings from the model
    full_dataset = ImageTrainDataset(TRAIN_DATA_FOLDER, train_data, transforms=val_transforms)
    loader = DataLoader(
        full_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.workers,
        pin_memory=True,
        drop_last=False,
    )

    features, targets = get_embeddings(model, loader)
    plot_tsne(features, targets)

wandb.finish()
