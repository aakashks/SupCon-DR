from md_clr import *
import pandas as pd
import timm
from torch.utils.data import DataLoader

import wandb

run = wandb.init(
    project="aml",
    dir=OUTPUT_FOLDER,
    config={
        k: v for k, v in CFG.__dict__.items() if not k.startswith('__')}
)

# train_data = pd.read_csv(os.path.join(DATA_FOLDER, 'trainLabels.csv'))
train_data = pd.read_csv(os.path.join(DATA_FOLDER, 'trainLabels_cropped.csv')).sample(frac=1).reset_index(drop=True)

# remove all images from the csv if they are not in the folder
lst = map(lambda x: x[:-5], os.listdir(TRAIN_DATA_FOLDER))
train_data = train_data[train_data.image.isin(lst)]

# take only 100 samples from each class
train_data = train_data.groupby('level').head(CFG.samples_per_class).reset_index(drop=True)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

def evaluate_model(cfg, model, data_loader, epoch=-1):

    model.eval()

    targets = []
    predictions = []

    total_len = len(data_loader)
    tk0 = tqdm(enumerate(data_loader), total=total_len)

    with torch.no_grad():
        for step, (images, labels) in tk0:
            images = images.to(device)
            target = labels.to(device)

            logits = model(images)

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


def train_epoch(cfg, model, train_loader, loss_criterion, optimizer, scheduler, epoch):
    loss_fn = loss_criterion

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

        logits = model(images)
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


def create_model():
    model = timm.create_model(CFG.model_name, num_classes=NUM_CLASSES, pretrained=True)

    # freeze the initial layers
    freeze_initial_layers(model, freeze_up_to_layer=CFG.frozen_layers)
    return model.to(device)


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
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6,
                                                           T_max=CFG.epochs * len(train_loader))

    loss_criterion = nn.CrossEntropyLoss()

    # TRAIN FOLD
    best_score = 0

    wandb.run.tags = [f"fold_{FOLD}"]

    for epoch in range(0, CFG.epochs):
        train_loss, train_lr, train_auc, train_accuracy, train_precision = train_epoch(CFG, model, train_loader,
                                                                                       loss_criterion, optimizer,
                                                                                       scheduler, epoch)

        val_loss, val_auc, val_accuracy, val_precision = evaluate_model(CFG, model, valid_loader, loss_criterion, epoch)

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
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'best_model_fold_{FOLD}.pth'))

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

    features, targets = get_embeddings(nn.Sequential(*list(model.children())[:-1]), loader)
    plot_tsne(features, targets)

wandb.finish()
