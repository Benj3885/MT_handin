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


def load_model(model, weights_path, device):
    ckpt = torch.load(weights_path) #checkpoint))

    model = VQVAE()

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
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--noise', action='store_true')
    args = parser.parse_args()


    ckpt = torch.load(args.weights, map_location=torch.device(device))
    model = VQVAE()
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()


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
            transforms.Resize((256, 256)),
        ]
    )


    to_pil = transforms.ToPILImage()
    for image in tqdm(images):
        img_in = Image.open(image)
        
        tensor_in = transform(img_in).unsqueeze(0).to(device)




        quant_t, quant_b, diff, id_t, id_b = model.encode(tensor_in)

        if args.noise:
            std = 5
            quant_t, quant_b, diff, id_t, id_b = model.encode(tensor_in, std)
        else:
            quant_t, quant_b, diff, id_t, id_b = model.encode(tensor_in)

        tensor_out = model.decode(quant_t, quant_b) #model(tensor_in)
        print(quant_t.size())
        print((quant_b.size()))


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

