from dataclasses import dataclass
from multiprocessing.spawn import prepare 
from pathlib import Path
from xml.sax.saxutils import prepare_input_source
import cv2
from PIL import Image
import argparse
from cv2 import dilate
import numpy as np
from yaml import load
from MiVOS.util.palette import pal_color_map
from MiVOS.interact.interactive_utils import overlay_davis
import os
from time import perf_counter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs_path", type=Path, required=True)
    parser.add_argument("--src_masks_path", type=Path, required=True)
    parser.add_argument("--out_masks_path", type=Path, required=True)

    parser.add_argument(
        "--src_mode", type=str, required=True, help="split (omnimattes) / together"
        )
    parser.add_argument(
        "--binary_threshold", type=float, default=0.25, help="threshold for omnimattes binarization"
        )

    parser.add_argument(
        "--out_mode", type=str, required=True, help="split (omnimattes) / together"
        )

    parser.add_argument(
        "--dilate_size", type=int, help="threshold for omnimattes binarization"
        )

    args = parser.parse_args()
    return args


@dataclass
class MaskerConfig:
    """Data container for Masker class"""
    imgs_path: Path
    src_masks_path: Path
    out_masks_path: Path 
    src_mode: str
    out_mode: str
    binary_threshold: float
    dilate_size: int

    @classmethod
    def load(cls, args):
        return cls(**args)


class Masker:
    def __init__(self, config: MaskerConfig):
        self.config = config
        self.idx = 0
        self.video_len = 0
        self.src_imgs = []
        self.src_masks = []
        self.out_masks = None
        self.out_mask_vis = None
        self.out_img_vis = None
        self.window_name = None
        self.font = None
        self.finished = False
        self.num_insances = 0
        self.colors = None
        self.colors_l = None
        self.colors_counts = 0
        self.pallete = 0
        self.instances_masks = None
        self.stacked_imgs = None
        self.stacked_masks = None
        self.selected_mask = None
        self.idx_text = ""
        self.text_color = (None, None, None)

    
    def load(self):
        """
        Load images and resize them to matching size

        Loading and resizing takes the most time.
        I menaged to speed up prepare_instances(), so now
        loading takes the most time and is a bootleneck.
        """
        src_imgs_pl = sorted(list(self.config.imgs_path.glob("*.png")))
        src_imgs = [cv2.imread(str(img_p)) for img_p in src_imgs_pl]

        if self.config.src_mode == "together":
            src_masks_pl = sorted(list(self.config.src_masks_path.glob("*.png")))
            src_masks = [cv2.imread(str(mask_p)) for mask_p in src_masks_pl]
        elif self.config.src_mode == "split":
            src_masks = self.load_split()

        min_l = min(len(src_imgs), len(src_masks))
        min_H = min(src_imgs[0].shape[0], src_masks[0].shape[0])
        min_W = min(src_imgs[0].shape[1], src_masks[0].shape[1])
        
        adj_imgs = []
        adj_masks = []
        for img, mask in zip(src_imgs[:min_l], src_masks[:min_l]):
            adj_imgs.append(cv2.resize(img, (min_W, min_H),interpolation=cv2.INTER_NEAREST))
            adj_masks.append(cv2.resize(mask, (min_W, min_H), interpolation=cv2.INTER_NEAREST))

        self.src_imgs = adj_imgs
        self.src_masks = adj_masks
        self.video_len = min_l


    def load_split(self):
        """
        Load masks that has been split into separate layers (omnimatte results)
        """
        src_masks = []
        # load masks in reverted order, as we want to start blending from background
        # mask_dirs = sorted([x for x in self.config.src_masks_path.iterdir() if x.is_dir() and str(x).split('/')[-1].isnumeric()])
        layers_num = len(list(self.config.src_masks_path.glob('0001_*_l*')))
        
        mask_dirs = [sorted(list(self.config.src_masks_path.glob(f"*_l{i}.*"))) for i in range(layers_num)]
        for idx, mask_dir in reversed(list(enumerate(mask_dirs))):
            # src_masks_pl = mask_dir#sorted(list(mask_dir.glob("*.png")))
            masks = []
            for mask_p in mask_dir:
                mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
                alpha = mask[:,:,3]
                alpha = np.repeat(alpha[..., np.newaxis], 3, axis=2)
                bin_mask = np.where(alpha >= (255*self.config.binary_threshold),idx,0)
                masks.append(bin_mask)

            if len(src_masks) < 1:
                src_masks = [np.zeros(masks[0].shape)]*len(masks)
            for idx, mask in enumerate(masks):
                src_masks[idx] = np.where(mask!=0, mask, src_masks[idx])

        return src_masks


    def prepare_instances(self):
        """
        Processes input data into operable format.

        Operating on RGB might be hard.
        We map each color from input mask into number of instance.
        This way it's easy to compare masks etc.
        This implementation should be optimized, but as we use it once,
        we can wait ~2s till data is mapped. 
        We are operating lineary on f.e (, 5619712), so it will take time. 


        inputs: stacked_imgs and masks [T,H,W,C] (0, 255)
        out: instance_masks [T, H, W] (0, num_instances)

        """
        # stacks of [T, H, W, C]
        self.stacked_imgs = np.stack(self.src_imgs)
        self.stacked_masks = np.stack(self.src_masks)

        # reshape to [-1(num_pixels),C], split into pixels and map RGB to bytes to get 1D array.
        # This should be optimizable, takes 1s
        byter = lambda t: t.tobytes()
        pixels = np.array([byter(pixel) for pixel in self.stacked_masks.reshape(-1, self.stacked_masks.shape[-1])])
        
        colors, self.colors_counts = np.unique(pixels, return_counts=True)
        
        # prepare mapping of rgb into instance number 
        self.colors = {RGB:num_instance for (num_instance, RGB) in enumerate(colors)}
        self.colors_l = [rgb for rgb in self.colors]
        self.num_insances = len(self.colors_l)

        # remap instances numbers into [T, H, W] and prepare blank output array 
        self.instances_masks = np.array([self.colors[rgb] for rgb in pixels], dtype=np.uint8).reshape(self.stacked_masks.shape[:-1])
        self.out_masks = self.instances_masks.clip(0, 0)


    def prepare_vis(self):
        """
        Prepares visualisation for current frame

        The visualisation is an image consisting of 2 images.
        1st is source image with overlayed instances, for easier mask selection.
        2nd is actual mask result, which will be saved if user finishes.
        """
        # Copy data representation for current frame
        out_img_vis = self.stacked_imgs[self.idx]
        cur_instances = self.instances_masks[self.idx].copy()

        #self.out_mask_vis = np.repeat(cur_mask[..., np.newaxis]*50, 3, axis=2)

        # colorize instances from [0, num_instances] to davis palette (RGBs) 
        cur_instances_img = Image.fromarray(cur_instances).convert("P")
        cur_instances_img.putpalette(self.pallete)
        cur_instances_img = cur_instances_img.convert("RGB")

        out_img_vis_rgb = cv2.cvtColor(out_img_vis, cv2.COLOR_BGR2RGB)
        out_img_vis = overlay_davis(out_img_vis_rgb, cur_instances, self.pallete)
        self.out_img_vis = cv2.cvtColor(out_img_vis, cv2.COLOR_RGB2BGR)

        # self.out_mask_vis = cv2.cvtColor(np.asarray(cur_instances_img,dtype=np.uint8), cv2.COLOR_RGB2BGR)

        cur_mask_img = self.prepare_out_img(self.idx)

        self.out_mask_vis = cv2.cvtColor(np.asarray(cur_mask_img,dtype=np.uint8), cv2.COLOR_RGB2BGR)
    

    def prepare_out_img(self, idx):
        cur_out_mask = self.out_masks[idx].copy()
        if self.config.dilate_size:
                cur_out_mask = self.dilate(cur_out_mask)
        cur_mask_img = Image.fromarray(cur_out_mask).convert("P")
        cur_mask_img.putpalette(self.pallete)
        cur_mask_img = cur_mask_img.convert("RGB")

        return cur_mask_img


    def prepare_out_split_imgs(self, idx):
        out_split_imgs = []
        cur_out_mask = self.out_masks[idx].copy()
        for num_instance in range(1, self.num_insances):
            split_mask = np.zeros(cur_out_mask.shape)
            split_mask = np.where(cur_out_mask == num_instance, 255, split_mask).astype(np.uint8)
            split_mask = np.repeat(split_mask[..., np.newaxis], 3, axis=2)
            if self.config.dilate_size:
                split_mask = self.dilate(split_mask)
            split_mask_img = Image.fromarray(split_mask)
            out_split_imgs.append(split_mask_img)

        return out_split_imgs

    def dilate(self, dilated_mask):
        dilation_shape = cv2.MORPH_ELLIPSE
        dilatation_size = self.config.dilate_size
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        return cv2.dilate(dilated_mask, element)

    def visualize(self):
        # print(f"{self.out_img_vis.shape}, {self.out_mask_vis.shape}")
        # print(f"{self.out_img_vis.dtype}, {self.out_mask_vis.dtype}")
        vis_img = cv2.vconcat([self.out_img_vis, self.out_mask_vis])
        cv2.putText(vis_img, self.idx_text, (5,20), self.font, 0.75, self.text_color, 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, vis_img)    


    def prepare_gui(self):
        self.window_name = "Masker"
        print(f"Setting up {self.window_name}...")

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.pallete = pal_color_map()

        def select_mask(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                # if in bounds of RGB image
                if y <= self.instances_masks[self.idx].shape[0] and x <= self.instances_masks[self.idx].shape[1]:
                    # remove last selection
                    if self.selected_mask is not None:
                        self.out_masks = np.where(self.out_masks == 10, 0, self.out_masks)

                    # get number of selected instance and swap all it's occurances to 10 (gui supports up to 10 output layers)
                    self.selected_mask = self.instances_masks[self.idx, y, x]
                    self.out_masks = np.where(self.instances_masks == self.selected_mask, 10, self.out_masks)

        cv2.setMouseCallback(self.window_name, select_mask)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (184, 3, 255)


    def prepare_text(self):
        idx_text = "Frame: " + str(self.idx) + "/" + str(self.video_len-1)
        self.idx_text = idx_text


    def handle_control(self):
        k = cv2.waitKey(33)
                                    # Usage
        if k == ord('q'):           # q to quit
            self.finished = True
        elif k == -1:               # default
            return 1
        elif k == ord('d'):         # next frame
            self.idx += 1
        elif k == ord('a'):         # previous frame
            self.idx -= 1
        elif self.selected_mask is not None and (k >= ord('0') and k <= ord('9')):
            self.update_results(chr(k))


    def update_results(self, selected_number):
        self.out_masks = np.where(self.instances_masks == self.selected_mask, int(selected_number), self.out_masks)
    

    def save(self):
        print("Do you want to save?\tyes / no")
        while(True):
            user_answer = input()
            if user_answer == "yes":
                break
            elif user_answer == "no":
                return
            else:
                print("Please, try again\t yes / no")

        self.config.out_masks_path.mkdir(parents=True,exist_ok=True)

        if self.config.out_mode == "split":
            layers_dirs = [self.config.out_masks_path / f"{num_layer:02}" for num_layer in range(1, self.num_insances)]
            for dir in layers_dirs:
                dir.mkdir(parents=True, exist_ok=True)
        
        for idx in range(self.video_len):
            if self.config.out_mode == "together":
                out_path = self.config.out_masks_path / f"{idx:04}.png"
                out_img = self.prepare_out_img(idx)
                out_img.save(out_path)

            elif self.config.out_mode == "split":
                out_imgs = self.prepare_out_split_imgs(idx)
                for layer_num, out_img in enumerate(out_imgs):
                    out_path = layers_dirs[layer_num] / f"{idx:04}.png"
                    out_img.save(out_path)


    def run(self):
        self.load()
        self.prepare_instances()
        self.prepare_gui()
        while(True):
            self.idx_limiter()
            self.prepare_text()
            self.prepare_vis()
            self.visualize()
            self.handle_control()

            if self.finished:
                break
        
        cv2.destroyAllWindows()
        self.save()

    def idx_limiter(self):
        if self.idx >= self.video_len:
            self.idx = self.video_len - 1
        elif self.idx < 0:
            self.idx = 0


def main():
    args = parse_args()
    config = MaskerConfig.load(vars(args))
    masker = Masker(config)
    masker.run()


if __name__ == "__main__":
    main()
    