import torch

class CFG:
    seed = 42
    
    # currently only validation 
    N_folds = 5
    train_folds = [0, ]  # [0,1,2,3,4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    apex = True  # use half precision
    workers = 16

    model_name = "resnet50.a1_in1k"
    epochs = 20
    cropped = True
    # weights =  torch.tensor([0.206119, 0.793881],dtype=torch.float32)

    clip_val = 1000.
    batch_size = 64
    # gradient_accumulation_steps = 1

    lr = 5e-3
    weight_decay = 1e-2

    resolution = 224
    samples_per_class = 1000
    frozen_layers = 0