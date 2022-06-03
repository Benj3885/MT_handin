import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
import mlflow


def train(args, epoch, loader, model, optimizer, device, val_loader):

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.
    n = 0
    total_correct = 0.

    for i, (top, bottom, label) in enumerate(pbar := tqdm(loader)):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        total_correct += correct.sum()
        n += target.numel()
        acc = correct.sum() / target.numel()

        pbar.set_description(
            (
                f'epoch: {epoch + 1}; avg loss: {total_loss / n:.5f}; '
                f'avg acc: {total_correct/n:.5f}; acc: {acc:.5f}'
            )
        )

    val_total_loss = 0
    val_n = 0

    with torch.no_grad():
        for i, (top, bottom, label) in enumerate(pbar := tqdm(val_loader)):
            top = top.to(device)

            if args.hier == 'top':
                target = top
                out, _ = model(top)

            elif args.hier == 'bottom':
                bottom = bottom.to(device)
                target = bottom
                out, _ = model(bottom, condition=top)

            loss = criterion(out, target)
            val_total_loss += loss.item()

            _, pred = out.max(1)
            correct = (pred == target).float()
            val_n += target.numel()
            accuracy = correct.sum() / target.numel()

            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; avg loss: {val_total_loss / val_n:.5f}; '
                    f'avg acc: {accuracy:.5f}'
                )
            )

    mlflow.log_metric('train_loss', total_loss / n)
    mlflow.log_metric('val_loss', val_total_loss / val_n)


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--off_set', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    os.makedirs(f"out/{args.out}", exist_ok=True)
    
    f = open(f"out/{args.out}/args.txt", "a")

    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

    mlflow.set_tracking_uri("file:/home/bjeh/BJEH/mlflow_pixelsnail/mlruns")

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    val_dataset = LMDBDataset(args.val_path)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )


    if args.hier == 'top':
        model = PixelSNAIL(
            [args.size//8, args.size//8],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [args.size//4, args.size//4],
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

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, device, val_loader)
        if i % args.save_interval == 0:
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f"out/{args.out}/pixelsnail_{args.hier}_{str(i + 1 + args.off_set).zfill(3)}.pt",
            )
