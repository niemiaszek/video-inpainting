import numpy as np
import os
import cv2
import glob
from pathlib import Path
import argparse
from PIL import Image

def load_images(src_path, out_path, mod_w, mod_h, resizing_mode):
    out_path.mkdir()
    fnames = sorted(glob.glob(os.path.join(src_path, '*.jpg')))
    if len(fnames) == 0:
        fnames = sorted(glob.glob(os.path.join(src_path, '*.png')))
    for i, fname in enumerate(fnames):
        image = Image.open(fname).convert('RGB')
        image = image.resize((mod_w, mod_h), resizing_mode)
        out_path_img = out_path / f"{i:04}.png"
        image.save(out_path_img)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--mod_w", type=int, required=True)
    parser.add_argument("--mod_h", type=int, required=True)
    parser.add_argument("--resizing_mask", type=int, help="pass 1 if resizing binary mask to keep it binary", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    resizing_mode = Image.BICUBIC
    if args.resizing_mask:
        resizing_mode = Image.NEAREST

    load_images(args.imgs_path, args.out_path, args.mod_w, args.mod_h, resizing_mode)

