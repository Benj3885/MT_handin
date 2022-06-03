import argparse
from cmath import inf
import os
import torch

import torch
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from vqvae import VQVAE
from glob import glob

def load_model(model, checkpoint, device):
    ckpt = torch.load(checkpoint)
    model = VQVAE()
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

def out_tensor_to_img(tensor_out):
    tensor_out = torch.squeeze(tensor_out)
    grid = make_grid(tensor_out, normalize=True, value_range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae', type=str, default='generators/vq_vae_2_pytorch/out/size512_defects_all/vqvae_161.pt')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--in_dir', type=str, default='/home/bjeh/BJEH/datasets/CAM3/Visible/randomized_dataset/train/')
    parser.add_argument('--out_dir', type=str, default='/home/bjeh/BJEH/CAM3/defects_recreated/')

    args = parser.parse_args()
    model_vqvae = load_model('vqvae', args.vqvae, device)

    transform_sample = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    os.makedirs(args.out_dir)
    paths = glob(args.in_dir + f'**/*.jpg', recursive=True)
    for path in tqdm(paths):
        fn = os.path.basename(path)
        img_in = Image.open(path)

        tensor = transform_sample(img_in).unsqueeze(0).to(device)
        model_out, _ = model_vqvae(tensor)
        model_out = model_out.clamp(-1, 1)
        img_out_rgb = out_tensor_to_img(model_out[0])
        img_out = img_out_rgb.convert('L')
        img_out.save(args.out_dir + fn)

    f = open(args.out_dir + "info.txt", "a")
    f.write(f"In_path: {args.in_dir}\n")
    f.write(f"Vqvae: {args.vqvae}\n")