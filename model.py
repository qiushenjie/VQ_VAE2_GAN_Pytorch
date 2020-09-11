import torch
from torch import nn
from base.mobilenetv2 import mobilenetv2
from vqvae import VQVAE
# from vqvae512 import VQVAE

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = VQVAE()

    def forward(self, x):
        dec, diff = self.generator(x)
        return dec, diff


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = mobilenetv2()

    def forward(self, x):
        output = self.discriminator(x)
        return output


if __name__ == '__main__':
    model = Generator()
    x = torch.Tensor(1, 2, 512, 512)
    output, diff = model(x)
    print(output.shape)
    print(diff)