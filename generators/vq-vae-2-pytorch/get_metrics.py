import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from tqdm import tqdm

from vqvae import VQVAE
import distributed as dist
import mlflow
from glob import glob


def get_metrics(epoch, loader, model, val_loader):
    criterion = nn.L1Loss()
    
    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0
    total_latent = 0
    total_loss = 0

    for i, (img, label) in enumerate(pbar := tqdm(loader)):
        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]

        mse_sum += part_mse_sum
        mse_n += part_mse_n
        total_latent += latent_loss.item() * latent_loss_weight
        total_loss += loss.item()

        pbar.set_description(
            (
                f"epoch: {epoch + 1}; l1: {recon_loss.item():.5f}; "
                f"latent: {total_latent / mse_n:.3f}; avg l1: {mse_sum / mse_n:.5f}; "
            )
        )


    val_mse_sum = 0
    val_mse_n = 0
    val_total_latent = 0
    val_total_loss = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(pbar := tqdm(val_loader)):
            img = img.to(device)

            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss

            part_mse_sum = recon_loss.item() * img.shape[0]
            part_mse_n = img.shape[0]
            
            val_mse_sum += part_mse_sum
            val_mse_n += part_mse_n
            val_total_latent += latent_loss.item() * latent_loss_weight
            val_total_loss += loss.item()

            pbar.set_description(
                (
                    f"epoch: {epoch + 1}; l1: {recon_loss.item():.5f}; "
                    f"latent: {val_total_latent / val_mse_n:.3f}; avg l1: {val_mse_sum / val_mse_n:.5f}; "
                )
            )


    mlflow.log_metric('train_loss_mse', mse_sum / mse_n)
    mlflow.log_metric('train_loss_latent', total_latent / mse_n)
    mlflow.log_metric('train_loss_total', total_loss / mse_n)
    mlflow.log_metric('val_loss_mse', val_mse_sum / val_mse_n)
    mlflow.log_metric('val_loss_latent', val_total_latent / val_mse_n)
    mlflow.log_metric('val_loss_total', val_total_loss / val_mse_n)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ckpt_dir", type=str, default="out/size512")
    parser.add_argument("--train_path", type=str, default='/Data/Real/CAM3/Good/current_version/train')
    parser.add_argument("--val_path", type=str, default='/Data/Real/CAM3/Good/current_version/validation/C')

    args = parser.parse_args()    

    mlflow.set_tracking_uri("file:/home/bjeh/BJEH/mlflow_vqvae_metrics/mlruns")
    

    for arg in vars(args):
        mlflow.log_param(arg, getattr(args, arg))

    print(args)

    device = "cuda:0"

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.train_path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=2)

    val_dataset = datasets.ImageFolder(args.val_path, transform=transform)
    val_sampler = dist.data_sampler(val_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, sampler=val_sampler, num_workers=2)

    model = VQVAE()

    ckpt_paths = glob(f'{args.ckpt_dir}/*.pt', recursive=True)
    ckpt_paths.sort()

    for i, ckpt_path in enumerate(ckpt_paths):
        model.load_state_dict(torch.load(ckpt_path))
        model.to(device).eval()
        get_metrics(i, loader, model, val_loader)
