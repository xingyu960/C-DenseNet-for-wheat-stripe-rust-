from torch import nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CBAM_Module(nn.Module):

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention module
        module_input = x
        avg_out = self.mlp(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        x = avg_out + max_out
        x = self.sigmoid(x)

        # Spatial attention module
        x = module_input * x
        module_input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = module_input * x
        return x

if __name__ == "__main__":
    CBAM_Module(in_planes=256)