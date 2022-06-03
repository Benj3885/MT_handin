import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
import distributed as dist
import mlflow


def train(epoch, loader, model, optimizer, device, args, val_loader):

    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    
    latent_loss_weight = 0.25
    sample_size = 10

    mse_sum = 0
    mse_n = 0
    total_latent = 0
    total_loss = 0

    for i, (img, label) in enumerate(pbar := tqdm(loader)):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        

        mse_sum += part_mse_sum
        mse_n += part_mse_n
        total_latent += latent_loss.item() * latent_loss_weight
        total_loss += loss.item()

        lr = optimizer.param_groups[0]["lr"]

        pbar.set_description(
            (
                f"epoch: {epoch + 1}; l1: {recon_loss.item():.5f}; "
                f"latent: {total_latent / mse_n:.3f}; avg l1: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 200 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"out/{args.out}/samples/{str(epoch + 1 + args.off_set).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                value_range=(-1, 1),
            )

            model.train()


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

            lr = optimizer.param_groups[0]["lr"]

            pbar.set_description(
                (
                    f"epoch: {epoch + 1}; l1: {recon_loss.item():.5f}; "
                    f"latent: {val_total_latent / val_mse_n:.3f}; avg l1: {val_mse_sum / val_mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )


    mlflow.log_metric('train_loss_mse', mse_sum / mse_n)
    mlflow.log_metric('train_loss_latent', total_latent / mse_n)
    mlflow.log_metric('train_loss_total', total_loss / mse_n)
    mlflow.log_metric('val_loss_mse', val_mse_sum / val_mse_n)
    mlflow.log_metric('val_loss_latent', val_total_latent / val_mse_n)
    mlflow.log_metric('val_loss_total', val_total_loss / val_mse_n)


def main(args):
    # Logging 

    mlflow.set_tracking_uri("file:/home/bjeh/BJEH/mlflow_vqvae/mlruns")

    device = "cuda:0"

    args.distributed = args.n_gpu > 1

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=args.batch // args.n_gpu, sampler=sampler, num_workers=2
    )

    val_dataset = datasets.ImageFolder(args.val_path, transform=transform)
    val_sampler = dist.data_sampler(val_dataset, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch // args.n_gpu, sampler=val_sampler, num_workers=2
    )

    
    model = VQVAE()
    if args.ckpt != None:
        model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    if args.distributed:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epoch):
        train(i, loader, model, optimizer, device, args, val_loader)

        if i % args.save_interval == 0:
            torch.save(model.state_dict(), f"out/{args.out}/vqvae_{str(i + 1 + args.off_set).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--off_set", type=int, default=0)
    parser.add_argument("--val_path", type=str, default='/Data/Real/CAM3/Good/current_version/validation/C')
    parser.add_argument("--sched", type=str)
    parser.add_argument("--out", type=str)

    parser.add_argument("path", type=str)

    args = parser.parse_args()    

    os.makedirs(f"out/{args.out}/samples", exist_ok=True)
    
    f = open(f"out/{args.out}/args.txt", "a")

    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

    print(args)

    main(args)
