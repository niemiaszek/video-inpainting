from pathlib import Path
import pathlib
import cv2
import numpy as np
from PIL import Image
import argparse

from yaml import parse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs_path", type=Path, required=True)
    parser.add_argument("--masks_path", type=Path, required=True)
    parser.add_argument("--bg_path", type=Path, required=True)
    parser.add_argument("--frame_rate", type=float, default=24)
    parser.add_argument("--output_path", type=Path, default=pathlib.Path('output_res'))
    args = parser.parse_args()

    return args


def switch_img(og_p, bg_p, m_p):
    #fg = mask
    #bg = upscaled fill
    #og = png fhg
    img_bg = cv2.imread(bg_p)
    img_m = cv2.imread(m_p)
    img_og = cv2.imread(og_p)

    img_og_sh = img_og.shape[:2]


    if(img_m.shape[:2] != img_og.shape[:2]):
        img_m = cv2.resize(img_m, img_og_sh[::-1], interpolation=cv2.INTER_NEAREST)
    if(img_bg.shape[:2] != img_og.shape[:2]):
        img_bg = cv2.resize(img_bg, img_og_sh[::-1], interpolation=cv2.INTER_CUBIC)

    img_bg_rgb = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
    img_og_rgb = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    
    return np.where(img_m, img_bg_rgb, img_og_rgb)

def probe_shape(img_p):
    img_probe = cv2.imread(img_p)
    print(img_probe.shape)
    return img_probe.shape


if __name__ == "__main__":
    args = parse_args()

    og_pl = sorted(list(args.imgs_path.glob('*.png')))
    masks_pl = sorted(list(args.masks_path.glob('*.png')))
    bg_pl = sorted(list(args.bg_path.glob('*.png')))

    final_l = min(len(og_pl), len(masks_pl), len(bg_pl))
    og_pl = og_pl[:final_l]
    masks_pl = masks_pl[:final_l]
    bg_pl = bg_pl[:final_l]

    switched_imges = []
    for og_p, bg_p, m_p in zip(og_pl, bg_pl, masks_pl):
        switched_imges.append(switch_img(str(og_p), str(bg_p), str(m_p)))
    print(og_pl)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    shape = probe_shape(str(og_pl[0]))[:2]
    args.output_path.mkdir(exist_ok=True)
    final_path = args.output_path / 'switcher.avi'
    out = cv2.VideoWriter(str(final_path), fourcc, args.frame_rate, shape[::-1])
    for frame in switched_imges:
        rgb = frame[...,::-1].copy()
        out.write(rgb)
    out.release()

