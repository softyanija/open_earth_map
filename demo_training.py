import os
import time
import warnings
import numpy as np
import torch
import oem
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "OpenEarthMap_Mini"
    TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = 512
    N_CLASSES = 9
    LR = 0.001
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs"
    GRAPH_DIR = "graphs"
    LOG_DIR = "logs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(fns))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    train_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Rotate(),
            oem.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(IMG_SIZE),
        ],
    )

    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = oem.losses.JaccardLoss()

    max_score = 0
    valid_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")
        x = list(range(1,NUM_EPOCHS+1))

        train_logs = oem.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = oem.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )

        
        #tem_loss = valid_logs["Loss"]
        #print("tem_loss = {}".format(tem_loss))
        valid_losses += [valid_logs["Loss"]] 
        
        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="model2.pth",
                output_dir=OUTPUT_DIR,
            )

    plt.figure(figsize=(12,8))
    plt.xlabel("Epoc", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.grid()
    plt.plot(x,valid_losses)
    plt.savefig(GRAPH_DIR+"/graph_test.png")
    
    """
    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    ax.plot(x,valid_losses)
    
    plt.plot()
    """
    #plt.show

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
