import argparse
import os

import torch
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
import threading


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device, size):
    ckpt = torch.load(checkpoint)
    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [size//8, size//8],
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
            [size//4, size//4],
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


def out_tensor_to_img(tensor):
    tensor = torch.squeeze(tensor)
    grid = make_grid(tensor, normalize=True, value_range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)

def handle_samples(args, device, id):
    model_vqvae = load_model('vqvae', args.vqvae, device, args.size)
    model_top = load_model('pixelsnail_top', args.top, device, args.size)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device, args.size)

    top_sample = sample_model(model_top, device, args.batch, [args.size//8, args.size//8], args.temp)
    bottom_sample = sample_model(model_bottom, device, args.batch, [args.size//4, args.size//4], args.temp, condition=top_sample)

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)


    for i, decoded_sample in enumerate(decoded_sample):
        img_out_rgb = out_tensor_to_img(decoded_sample)
        img_out = img_out_rgb.convert('L')
        img_out.save(f"edit_out/{args.out_dir}/sample_" + str(i + (args.batch*id)) + ".jpg")

if __name__ == '__main__':
    device = 'cuda:'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str, default="out/size512/vqvae_011.pt")
    parser.add_argument('--top', type=str, default="out/pixelsnail512/pixelsnail_top_003.pt")
    parser.add_argument('--bottom', type=str, default="out/pixelsnail512/pixelsnail_bottom_003.pt")
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--out_dir', type=str, default="new_samples")
    parser.add_argument('--n_gpus', type=int, default=2)

    args = parser.parse_args()

    threads = []
    for id in range(2):
        thread = threading.Thread(target=handle_samples, args=(args, device+str(id), id,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
