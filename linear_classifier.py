from md_clr import *
from md_clr.supcon import *

run = wandb.init(
    project="hello-world",
    dir=OUTPUT_FOLDER,
    config={
        k: v for k, v in CFG.__dict__.items() if not k.startswith('__')}
)

get_train_data()

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
    if CFG.feature_extractor == "resnet":
        feature_extractor = timm.create_model(CFG.model_name, pretrained=False)
        feature_extractor.load_state_dict(torch.load(OUTPUT_FOLDER + 'tl_model.pth'))

    elif CFG.feature_extractor == "resnet_sclr":
        resnet = timm.create_model(CFG.model_name, num_classes=0, pretrained=False)
        feature_extractor = SupConModel(resnet)
        feature_extractor.load_state_dict(torch.load(OUTPUT_FOLDER + 'ckpt_epoch_8.pth'))
        
        # remove the projection head
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])

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

wandb.finish()
