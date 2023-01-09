import os
import time
import rasterio
import warnings
import numpy as np
import torch
import torchvision
import cv2
import oem
from pathlib import Path
from PIL import Image
import math

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "OpenEarthMap_Mini"
    PR_DIR = "png"
    os.makedirs(PR_DIR, exist_ok=True)
    TEST_LIST = os.path.join(OEM_DATA_DIR, "test.txt")

    N_CLASSES = 9
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PREDS_DIR = "predictions"
    os.makedirs(PREDS_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    test_fns = [str(f) for f in fns if f.name in np.loadtxt(TEST_LIST, dtype=str)]

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))

    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=None,
        testing=True,
    )

    network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    network = oem.utils.load_checkpoint(
        network,
        model_name="model.pth",
        model_dir="outputs",
    )

    save_fns = []

    network.eval().to(DEVICE)
    for test_fn in test_fns:
        img = Image.fromarray(oem.dataset.load_multiband(test_fn))
        
        w, h = img.size[:2]
        power_h = math.ceil(np.log2(h) / np.log2(2))
        power_w = math.ceil(np.log2(w) / np.log2(2))
        if 2**power_h != h or 2**power_w != w:
            img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
        img = np.array(img)

        # test time augmentation
        imgs = []
        imgs.append(img.copy())
        imgs.append(img[:, ::-1, :].copy())
        imgs.append(img[::-1, :, :].copy())
        imgs.append(img[::-1, ::-1, :].copy())

        input = torch.cat([torchvision.transforms.functional.to_tensor(x).unsqueeze(0) for x in imgs], dim=0).float().to(DEVICE)

        pred = []
        with torch.no_grad():
            msk = network(input) 
            msk = torch.softmax(msk[:, :, ...], dim=1)
            msk = msk.cpu().numpy()
            pred = (msk[0, :, :, :] + msk[1, :, :, ::-1] + msk[2, :, ::-1, :] + msk[3, :, ::-1, ::-1])/4

            pred = Image.fromarray(pred.argmax(axis=0).astype("uint8"))
            y_pr = pred.resize((w, h), resample=Image.NEAREST)

            filename = os.path.basename(test_fn).replace('tif','png')
            save_fn = os.path.join(PR_DIR, filename)
            y_pr.save(save_fn)
            save_fns.append(save_fn)
    
'''
    network.eval().to(DEVICE)
    for idx in range(len(test_fns)):
        img, fn = test_data[idx][0], test_data[idx][2]

        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

        fout = os.path.join(PREDS_DIR, fn.split("/")[-1])
        with rasterio.open(fn, "r") as src:
            profile = src.profile
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            with rasterio.open(fout, "w", **profile) as dst:
                for idx in src.indexes:
                    dst.write(prd[:, :, idx - 1], idx)
'''
