import argparse
from sqlalchemy import false
from tqdm import tqdm
import torch
from vqvae import VQVAE
from pixelsnail import PixelSNAIL
import coco
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
import threading
import os

@torch.no_grad()
def sample_box(model, device, batch, size, temperature, condition=None, extra=None, row_range=(1, 1), col_range = (1, 1), row_skip=1, col_skip=1, max=False):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0]), leave=False):
        if (extra != None and i < int(size[0]*row_range[0]) or i > int(size[0]*row_range[1] + 0.99) or (i % row_skip != 0 and i > 0)) or model == None:
            row[:, i, :] = extra[i, :]
            continue
        for j in range(size[1]):
            if (j < int(size[1]*col_range[0]) or j > int(size[1]*col_range[1] + 0.99) or (j % col_skip != 0 and j > 0)):
                row[:, i, j] = extra[i, j]
                continue
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            if max:
                sample = torch.argmax(prob, 1).squeeze(-1)
            else:
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
            [64, 64],
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
            [128, 128],
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

def out_tensor_to_img(tensor_out):
    tensor_out = torch.squeeze(tensor_out)
    grid = make_grid(tensor_out, normalize=True, value_range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)

def thread_handling(args, ids, device):

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    for id in tqdm(ids):
        img_by_id = dataset.get_img_by_id(id)
        fn = img_by_id['file_name']
        height = img_by_id['height']
        width = img_by_id['width']

        img_in = Image.open(args.in_dir + fn)

        tensor = transform_sample(img_in).unsqueeze(0).to(device)
        _, _, _, top, bot = model_vqvae.encode(tensor)

        for ann in dataset.get_anns_by_image_id(id):
            bbox = ann['bbox']
            height_scale = args.size / height
            width_scale = args.size / width
            bbox = [bbox[0] * width_scale, bbox[1] * height_scale, bbox[2] * width_scale, bbox[3] * height_scale]

            hor_range = (bbox[0] / args.size, (bbox[0] + bbox[2]) / args.size)
            ver_range = (bbox[1] / args.size, (bbox[1] + bbox[3]) / args.size)

            top_sample = sample_box(model_top, device, args.batch, [args.size // 8, args.size // 8], args.temp, extra=top[0], row_range=ver_range, col_range=hor_range, max=False)
            bot_sample = sample_box(model_bottom, device, args.batch, [args.size // 4, args.size // 4], args.temp, condition=top_sample, extra=bot[0], row_range=ver_range, col_range=hor_range, max=False)

        decoded_sample = model_vqvae.decode_code(top, bot_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)

        img_out_rgb = out_tensor_to_img(decoded_sample[0])
        img_out = img_out_rgb.convert('L')

        # Save images
        img_out.save(args.out_dir + fn)



if __name__ == '__main__':
    device = 'cuda:0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae', type=str, default='out/size512/vqvae_011.pt')
    parser.add_argument('--bottom', type=str, default='out/pixelsnail512/pixelsnail_bottom_003.pt')
    parser.add_argument('--top', type=str, default='out/pixelsnail512/pixelsnail_top_003.pt')
    parser.add_argument('--in_dir', type=str, default='/home/bjeh/BJEH/datasets/CAM3/Visible/randomized_dataset/train/')
    parser.add_argument('--out_dir', type=str, default='/home/bjeh/BJEH/datasets/CAM3/defect_removed_1_0/')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=1)

    args = parser.parse_args()


    transform_sample = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    os.makedirs(args.out_dir)
    dataset = coco.COCO(filename=args.in_dir + "instances_default.json")

    ids = dataset.id_imgname_dict(0)
    ids = [x+1 for x in list(range(len(ids)))]
    ids1 = ids[:len(ids)//2]
    ids2 = ids[len(ids)//2:]
    
    threads = []
    thread1 = threading.Thread(target=thread_handling, args=(args, ids1, f'cuda:' + str(0)))
    thread1.start()
    threads.append(thread1)
    thread2 = threading.Thread(target=thread_handling, args=(args, ids2, f'cuda:' + str(1)))
    thread2.start()
    threads.append(thread2)

    for thread in threads:
        thread.join()
