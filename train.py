from functions.dataHandling import get_dataset
from functions.visualize import plot_grid_samples_tensor
from models.vit import VIT

import torch
from torch.utils.data import DataLoader

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


def train_VIT(settings):

    # Tensorboard for logging
    writer = SummaryWriter()
    # Tensorboard doesnt support dicts in hparam dict so lets unpack

    # Print settings and info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(str(settings))
    print(f"Device: {device}" + "\n")

    # Loading dataset
    train, test, input_shape, channels = get_dataset(settings["dataset"], print_stats=True)
    train_var = (train.data /255.0).var()

    dataloader_train = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    dataloader_test = DataLoader(test, batch_size=256, num_workers=1, pin_memory=True)

    # Setting up model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] = channels
    model_settings["input_shape"] = input_shape
    model_settings["device"] = device

    model = VIT(model_settings).to(device)
    if settings["print_debug"]:
        print(summary(VIT(), input_size=(256, 1,32,32)))

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], amsgrad=False, weight_decay=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda" ,enabled=True)

    # Training loop
    train_losses, test_losses = [], []
    for epoch in range(settings["max_epochs"]):
        train_losses_epoch = []
        print(f"Epoch: {epoch}/{settings["max_epochs"]}")
        
        # Training
        model.train()
        for batch_i, (x_train, y_train) in enumerate(dataloader_train):
            x_train, y_train = x_train.to(device), y_train.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                pred = model(x_train)
                loss = loss_function(pred, y_train)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses_epoch.append(loss.item())
            optimizer.zero_grad()

            
        print(f"Train loss: {sum(train_losses_epoch) / len(train_losses_epoch)}")
        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        writer.add_scalar("Loss/train", train_losses[-1], epoch)

        #  Early stopping
        epoch_delta = settings["early_stopping_epochs"]
        if len(train_losses) > epoch_delta and max(train_losses[-epoch_delta:-1]) < train_losses[-1]:
            print("Early stopping")
            break
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_losses_epoch = []
            val_corrects = 0
            for x_test, y_test in dataloader_test:
                x_test, y_test = x_test.to(device), y_test.to(device)
                output = model(x_test)
                loss = loss_function(output, y_test)
                test_losses_epoch.append(loss.item())

                _, val_preds = torch.max(output, 1)
                val_corrects += torch.sum(val_preds == y_test)
            print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
            test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))

            print(f"Test acc: {val_corrects / len(dataloader_test.dataset)}")
            
            writer.add_scalar("Loss/test", test_losses[-1], epoch)
            writer.add_scalar("ACC/test", val_corrects / len(dataloader_test.dataset), epoch)

              

    # Save trained model to disk
    if settings["save_model"]:
        import os
        path = f"models/saved_models/{settings["dataset"]}/"
        os.makedirs(path, exist_ok = True) 
        torch.save(model.state_dict(), path + "model.pt")