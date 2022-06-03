import torch
from PIL import Image
from cmath import inf
from generators.vq_vae_2_pytorch.vqvae import VQVAE
from generators.vq_vae_2_pytorch.pixelsnail import PixelSNAIL
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision import transforms

def vqvae_load_model(model, checkpoint, size):
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

    elif model == 'pixelsnail_bot':
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
    model.eval()

    return model

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None, extra=None, row_ratio=1.0, row_skip=1, col_skip=1):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in (pbar := tqdm(range(size[0]), leave=False)):
        pbar.set_description((f"sample latent representation"))
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

    for i in (pbar := tqdm(range(size[0]), leave=False)):
        pbar.set_description((f"sample latent representation"))
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


def out_tensor_to_img(tensor_out):
    tensor_out = torch.squeeze(tensor_out)
    grid = make_grid(tensor_out, normalize=True, value_range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


class vqvae_generator():
    def __init__(self, vqvae_ckpt, ps_bot_ckpt, size):
        self.vqvae = vqvae_load_model('vqvae', vqvae_ckpt, size=size)
        #self.ps_top = vqvae_load_model('pixelsnail_top', ps_top_ckpt, size)
        self.ps_bot = vqvae_load_model('pixelsnail_bot', ps_bot_ckpt, size)
        self.size=size
        self.device = 'cpu'

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def to(self, device):
        self.vqvae.to(device)
        self.ps_bot.to(device)
        self.device = device

    def augment(self, inp, n_outputs, bot_row_ratio=0, temp=1., row_interval=1, col_interval=1):
        tensor = self.transform(inp).unsqueeze(0).to(self.device)
        _, _, _, top, bot = self.vqvae.encode(tensor)

        #top = vqvae_sample_model(self.ps_top, self.device, n_outputs, [self.size//8, self.size//8], temp, extra=top)
        bot = sample_model(self.ps_bot, 
                            self.device, 
                            n_outputs, 
                            [self.size//4, self.size//4], 
                            temp, 
                            condition=top, 
                            extra=bot[0], 
                            row_ratio=bot_row_ratio,
                            row_skip=row_interval,
                            col_skip=col_interval
                            )

        new_imgs = self.vqvae.decode_code(top, bot)
        new_imgs = new_imgs.clamp(-1, 1)

        imgs_out = []
        for new_img in new_imgs:
            img_out_rgb = out_tensor_to_img(new_img)
            img_out = img_out_rgb.convert('L')
            imgs_out.append(img_out)
        return imgs_out
