"""CLI wrapper around RISE for batch generation of saliency maps."""

import argparse
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

from .rise import RISE
from .model import get_model

def overlay(img: Image.Image, sal: np.ndarray, alpha: float = 0.5):
    plt.imshow(img)
    plt.imshow(sal, cmap='jet', alpha=alpha)
    plt.axis('off')

def load_lookup(csv_path: Path, key_col="id_code", val_col="diagnosis"):
    df = pd.read_csv(csv_path)
    if key_col not in df.columns or val_col not in df.columns:
        raise KeyError(f"Expected columns '{key_col}' & '{val_col}'")
    return dict(zip(df[key_col].astype(str), df[val_col]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--images', type=str, required=True,
                        help='Path to directory with *.jpg/*.png files')
    parser.add_argument('--outdir', type=str, default='outputs/maps')
    parser.add_argument('--N', type=int, default=4000)
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to train.csv")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device).eval()

    Path(args.outdir).mkdir(exist_ok=True, parents=True)
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        # T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


    # explainer = RISE(model, input_size=(224,224), N=args.N, device=device)
    explainer = RISE(
        model,
        input_size=(224, 224),
        N=args.N,
        s=100,
        p=0.7,
        batch=500,
        device=device,
    )

    lookup = load_lookup(args.csv)
    for img_path in Path(args.images).glob('*'):
        # img_path = Path('data/train_retina/002c21358ce6.png')
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        sal = explainer.explain(img_tensor, target=lookup[str(img_path.name).split('.')[0]]).numpy()

        # save overlay
        plt.figure(figsize=(6,6))
        overlay(img, sal)
        out_file = Path(args.outdir) / f'{img_path.stem}_rise.png'
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    main()
