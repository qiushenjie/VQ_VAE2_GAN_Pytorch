import argparse
import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist
from utils.Dataset import txtDataset
from loss import GDLoss
from model import Generator, Discriminator

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

n_critic = 2
batch_size = 6
def train(epoch, loader, discriminator, generator, scheduler_D, scheduler_G, optimizer_D, optimizer_G, device):
    loader_d = tqdm(loader)
    if (epoch + 1) % n_critic == 0:
        loader_g = tqdm(loader)

    adversarial_loss = nn.BCEWithLogitsLoss()  # sigmoid
    pixelwise_loss = nn.L1Loss()
    gdloss = GDLoss()

    recon_loss_weight = 0.4
    latent_loss_weight = 0.2
    gradient_loss_weight = 0.4
    sample_size = batch_size

    mse_sum = 0
    mse_n = 0
    g_sum = 0
    g_n = 0

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    # ---------------------
    #  Train Discriminator
    # ---------------------
    for i, (img, label, label_path, class_name) in enumerate(loader_d):
        discriminator.zero_grad()

        valid = Variable(torch.Tensor(img.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.Tensor(img.shape[0], 1).fill_(0.0), requires_grad=False)

        img = img.to(device)
        valid = valid.to(device)
        fake = fake.to(device)
        label = label.to(device)

        gdloss.conv_x = gdloss.conv_x.to(device)
        gdloss.conv_y = gdloss.conv_y.to(device)

        vqvae2_out, latent_loss = generator(img)

        real_loss = adversarial_loss(discriminator(label), valid)
        fake_loss = adversarial_loss(discriminator(vqvae2_out), fake)

        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()

        if scheduler_D is not None:
            scheduler_D.step()
        optimizer_D.step()

        if dist.is_primary():
            lr = optimizer_D.param_groups[0]["lr"]

            loader_d.set_description(
                (
                    f"Discriminator epoch: {epoch + 1}; class loss: {d_loss.item():.5f};"
                    f"lr: {lr:.5f}"
                )
            )

    # ---------------------
    #  Train Generator
    # ---------------------
    if (epoch + 1) % n_critic == 0:
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        for i, (img, label, label_path, class_name) in enumerate(loader_g):
            generator.zero_grad()

            valid = Variable(torch.Tensor(img.shape[0], 1).fill_(1.0), requires_grad=False)

            img = img.to(device)
            valid = valid.to(device)
            label = label.to(device)

            gdloss.conv_x = gdloss.conv_x.to(device)
            gdloss.conv_y = gdloss.conv_y.to(device)

            vqvae2_out, latent_loss = generator(img)

            recon_loss = pixelwise_loss(vqvae2_out, label)
            gradient_loss = gdloss(vqvae2_out, label)
            gradient_loss = gradient_loss.mean()
            latent_loss = latent_loss.mean()
            g_loss = 0.1 * adversarial_loss(discriminator(vqvae2_out), valid) + \
                     0.9 * (recon_loss_weight * recon_loss + latent_loss_weight * latent_loss + gradient_loss_weight * gradient_loss)

            g_loss.backward()

            if scheduler_G is not None:
                scheduler_G.step()
            optimizer_G.step()

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
                lr = optimizer_G.param_groups[0]["lr"]

                loader_g.set_description(
                    (
                        f"Denerator epoch: {(epoch + 1) // n_critic + 1}; mse: {recon_loss.item():.5f}; "
                        f"latent: {latent_loss.item():.3f}; gradient: {g_sum / g_n:.5f}; avg mse: {mse_sum / mse_n:.5f}; "
                        f"lr: {lr:.5f}"
                    )
                )

            if i % 100 == 0:
                generator.eval()

                sample = img[:sample_size]
                label_sample = label[:sample_size]
                sample0 = sample[:, 0, :, :].unsqueeze(dim=1)
                sample1 = sample[:, 1, :, :].unsqueeze(dim=1)
                a = (sample1.data.cpu()).numpy()
                with torch.no_grad():
                    out, _ = generator(sample)

                utils.save_image(
                    torch.cat([sample0, sample1, label_sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                generator.train()


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

    txt_path = './data/train.txt'
    images_path = './data'
    labels_path = './data'

    dataset = txtDataset(txt_path, images_path, labels_path, transform=transform)

    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=batch_size // args.n_gpu, sampler=sampler, num_workers=16
    )

    # Initialize generator and discriminator
    DpretrainedPath = './checkpoint/vqvae2GAN_040.pt'
    GpretrainedPath = './checkpoint/vqvae_040.pt'

    discriminator = Discriminator()
    generator = Generator()
    if os.path.exists(DpretrainedPath):
        print('Loading model weights...')
        discriminator.load_state_dict(torch.load(DpretrainedPath)['discriminator'])
        print('done')
    if os.path.exists(GpretrainedPath):
        print('Loading model weights...')
        generator.load_state_dict(torch.load(GpretrainedPath))
        print('done')

    discriminator = discriminator.to(device)
    generator = generator.to(device)

    if args.distributed:
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr)
    scheduler_D = None
    scheduler_G = None
    if args.sched == "cycle":
        scheduler_D = CycleScheduler(
            optimizer_D,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

        scheduler_G = CycleScheduler(
            optimizer_G,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )


    for i in range(41, args.epoch):
        train(i, loader, discriminator, generator, scheduler_D, scheduler_G, optimizer_D, optimizer_G, device)

        if dist.is_primary():
            torch.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': optimizer_G.state_dict(),
                    'd_optimizer': optimizer_D.state_dict(),
                },
                f'checkpoint/vqvae2GAN_{str(i + 1).zfill(3)}.pt',
            )
            if (i + 1) % n_critic == 0:
                torch.save(generator.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


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
