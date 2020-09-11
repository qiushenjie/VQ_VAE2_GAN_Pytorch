import argparse
import sys
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
from utils.Dataset import txtDataset
from loss import GDLoss

batch_size = 24
def train(epoch, loader, model, optimizer, scheduler, device):
    # if dist.is_primary():
    #     loader = tqdm(loader)
    loader = tqdm(loader)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    gdloss = GDLoss()

    recon_loss_weight = 0.4
    latent_loss_weight = 0.2
    gradient_loss_weight = 0.4
    sample_size = batch_size

    mse_sum = 0
    mse_n = 0
    g_sum = 0
    g_n = 0

    for i, (img, label, label_path, class_name) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)
        label = label.to(device)
        gdloss.conv_x = gdloss.conv_x.to(device)
        gdloss.conv_y = gdloss.conv_y.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, label)
        gradient_loss = gdloss(out, label)
        gradient_loss = gradient_loss.mean()
        latent_loss = latent_loss.mean()
        loss = recon_loss_weight * recon_loss + latent_loss_weight * latent_loss + gradient_loss_weight * gradient_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        part_g_sum = gradient_loss.item() * img.shape[0]
        part_g_n = img.shape[0]
        g_comm = {"g_sum": part_g_sum, "g_n": part_g_n}
        g_comm = dist.all_gather(g_comm)

        for part in g_comm:
            g_sum += part["g_sum"]
            g_n += part["g_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; gradient: {g_sum / g_n:.5f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]
                label_sample = label[:sample_size]
                sample0 = sample[:, 0, :, :].unsqueeze(dim=1)
                sample1 = sample[:, 1, :, :].unsqueeze(dim=1)
                a = (sample1.data.cpu()).numpy()
                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample0, sample1, label_sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    normMean = [0.5]
    normStd = [0.5]

    normTransform = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([transforms.Resize(args.size),
                                         transforms.ToTensor(),
                                         normTransform,
                                         ])

    txt_path = 'datd/train.txt'
    images_path = '/data'
    labels_path = '/data'

    dataset = txtDataset(txt_path, images_path, labels_path, transform=transform)

    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=batch_size // args.n_gpu, sampler=sampler, num_workers=16
    )

    pretrainedPath = '/checkpoint/saved_model.pt'
    model = VQVAE()
    if os.path.exists(pretrainedPath):
        print('Loading model weights...')
        model.load_state_dict(torch.load(pretrainedPath))
        print('done')
    model = model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(93, args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str, default='cycle')
    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
