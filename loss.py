from torch import nn
import torch
import numpy as np

class GDLoss(nn.Module):  # gradient difference loss
    def __init__(self):
        super(GDLoss, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        conv_x = np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        conv_y = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.conv_x.weight = nn.Parameter(torch.FloatTensor(conv_x).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.FloatTensor(conv_y).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, outputs, labels):
        outputs_grd_x = self.conv_x(outputs)
        outputs_grd_y = self.conv_y(outputs)

        labels_grd_x = self.conv_x(labels)
        labels_grd_y = self.conv_y(labels)

        grid_diff_x = torch.norm(torch.abs(outputs_grd_x) - torch.abs(labels_grd_x), dim=1)
        grid_diff_y = torch.norm(torch.abs(outputs_grd_y) - torch.abs(labels_grd_y), dim=1)
        return grid_diff_x + grid_diff_y

if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).unsqueeze(0), requires_grad=True)
    loss = GDLoss()
    print(loss.conv_x.weight)
    z = loss.forward(x)
    print(z)
    z.backward()
    print(x.grad)
