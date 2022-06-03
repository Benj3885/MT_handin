import argparse
from cmath import inf
import os

import torch
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image, ImageChops
import numpy as np
import cv2
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from glob import glob
import random


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None, extra=None, row_ratio=1.0, row_skip=1, col_skip=1):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        if (extra != None and i < int(size[0]*row_ratio) or (i % row_skip != 0 and i > 0)) or model == None:
            row[:, i, :] = extra[i, :]
            continue
        for j in range(size[1]):
            if j % col_skip != 0 and j > 0:
                row[:, i, j] = extra[i, j]
                continue
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


@torch.no_grad()
def sample_box(model, device, batch, size, temperature, condition=None, extra=None, row_range=(1, 1), col_range = (1, 1), row_skip=1, col_skip=1):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        if (extra != None and i < int(size[0]*row_range[0]) or i > int(size[0]*row_range[1]) or (i % row_skip != 0 and i > 0)) or model == None:
            row[:, i, :] = extra[i, :]
            continue
        for j in range(size[1]):
            if (j < int(size[1]*col_range[0]) or j > int(size[1]*col_range[1]) or (j % col_skip != 0 and j > 0)):
                row[:, i, j] = extra[i, j]
                continue
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row

def load_model(model, checkpoint, image_size, device):
    ckpt = torch.load(checkpoint)
    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [image_size // 8, image_size // 8],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bot':
        model = PixelSNAIL(
            [image_size // 4, image_size // 4],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    return model


def get_concat_h(im1, im2, im3):
    dst = Image.new('RGB', (im1.width * 3, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width * 2, 0))
    return dst


def get_heatmap(img_comp):        
    img_comp_cv = np.array(img_comp)
    heatmap = cv2.applyColorMap(img_comp_cv, cv2.COLORMAP_JET)
    return Image.fromarray(heatmap[:, :, ::-1])


def out_tensor_to_img(tensor_out):
    tensor_out = torch.squeeze(tensor_out)
    grid = make_grid(tensor_out, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


if __name__ == '__main__':
    device = 'cuda:0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--vqvae', type=str, default=".pt")
    parser.add_argument('--top', type=str, default=".pt")
    parser.add_argument('--bottom', type=str, default=".pt")
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--size', type=int, default=512)

    args = parser.parse_args()
    model_vqvae = load_model('vqvae', args.vqvae, args.size, device)
    model_top = load_model('pixelsnail_top', args.top, args.size, device)
    model_bottom = load_model('pixelsnail_bot', args.bottom, args.size, device)

    transform_comp = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size))
        ]
    )
    transform_sample = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )


    paths = [".jpg"]
    random.shuffle(paths)
    for im_i, path in enumerate(paths):
        fn = os.path.basename(path)
        img_in = Image.open(path)
        img_in = transform_comp(img_in)

        print(path)

        tensor = transform_sample(img_in).unsqueeze(0).to(device)
        _, _, _, top, bot = model_vqvae.encode(tensor)


        torch.manual_seed(0)
        if True:
            #top = sample_model(model_top, device, args.batch, [args.size // 8, args.size // 8], args.temp, extra=top[0], row_ratio=1.)
            bot = sample_model(model_bottom, device, args.batch, [args.size // 4, args.size // 4], args.temp, condition=top[0].unsqueeze(0), extra=bot[0], row_ratio=0.5, row_skip=1, col_skip=1)
        else:
            row_range = (0.1, 0.25)
            col_range = (0.5, 0.75)
            #top = sample_box(model_top, device, args.batch, [args.size // 8, args.size // 8], args.temp, extra=top[0], row_range=row_range, col_range=col_range)
            bot = sample_box(model_bottom, device, args.batch, [args.size // 4, args.size // 4], args.temp, condition=top[0].unsqueeze(0), extra=bot[0], row_range=row_range, col_range=col_range, row_skip=1, col_skip=1)

        decoded_samples = model_vqvae.decode_code(top, bot)
        decoded_samples = decoded_samples.clamp(-1, 1)

        for i, decoded_sample in enumerate(decoded_samples):
            img_out_rgb = out_tensor_to_img(decoded_sample)
            img_out = img_out_rgb.convert('L')

            # Get comparisons (absolute difference)
            img_comp = get_heatmap(ImageChops.difference(img_in, img_out))

            # Concatenate images
            imgs = get_concat_h(img_in, img_out, img_comp)

            # Save images
            imgs.save("edit_out/" + str(im_i) + "_" + str(i) + os.path.basename(fn))

