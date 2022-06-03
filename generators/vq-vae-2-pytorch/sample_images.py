import argparse
import os
from sympy import arg

import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from tqdm import tqdm

from vqvae import VQVAE
from PIL import Image, ImageChops
import glob
import numpy as np
import cv2

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None, extra=None, row_ratio=1.0):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    if extra != None:
        row[:, :int(size[0]*row_ratio), :] = extra[:int(size[0]*row_ratio), :]

    for i in tqdm(range(size[0])):
        if extra != None and i < int(size[0]*row_ratio):
            continue
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device):
    ckpt = torch.load(checkpoint)

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [64, 64],
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

    model.load_state_dict(ckpt)
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


def out_tensor_to_img(tensor_out, transform):
    tensor_out = transform(tensor_out)
    tensor_out = torch.squeeze(tensor_out)
    grid = make_grid(tensor_out, normalize=True, range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)


    images = glob.glob("edit_in/*")

    files = glob.glob('edit_out/*')
    for f in files:
        os.remove(f)


    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    transform_ogsize = transforms.Compose(
        [
            transforms.Resize((892, 576)),
        ]
    )

    transform_scaled = transforms.Compose(
        [
            transforms.Resize((512, 256)),
        ]
    )


    to_pil = transforms.ToPILImage()
    for image in tqdm(images):
        img_in = Image.open(image)
        
        tensor_in = transform(img_in).unsqueeze(0).to(device)

        quant_t, quant_b, diff, id_t, id_b = model_vqvae.encode(tensor_in)

        top_sample = sample_model(model_top, device, args.batch, [99, 64], args.temp, extra=quant_t)
        bottom_sample = sample_model(model_bottom, device, args.batch, [198, 128], args.temp, condition=top_sample)


        tensor_out = model_vqvae.decode_code(top_sample, bottom_sample)

        img_out_rgb = out_tensor_to_img(tensor_out, transform_ogsize)
        img_out = img_out_rgb.convert('L')
        # Get comparisons (absolute difference)
        img_comp = get_heatmap(ImageChops.difference(img_in, img_out))

        # Concatenate images
        imgs = get_concat_h(img_in, img_out, img_comp)

        # Save images
        imgs.save("edit_out/" + os.path.basename(image))

        if args.scale:
            img_in_scaled = transform_scaled(img_in)
            img_out_scaled = transform_scaled(img_out)
            img_comp_scaled = get_heatmap(ImageChops.difference(img_in_scaled, img_out_scaled))
            imgs_scaled = get_concat_h(img_in_scaled, img_out_scaled, img_comp_scaled)
            imgs_scaled.save("edit_out/scaled_" + os.path.basename(image))

