"""
Compact CNN for MNIST deep ensembles.

Uses 3-channel input so the same model naturally handles:
  - MNIST digits (replicated to 3ch during loading)
  - Color-corrupted digits (colored backgrounds, tinted digits)
  - Native RGB OOD datasets (CIFAR-10, SVHN) without forced grayscale

Supports per-member diversity knobs: dropout before classifier, head init scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """Compact CNN for MNIST (28x28, 3-channel input).

    Architecture:
        Conv(3→32, 3) → BN → ReLU → Conv(32→32, 3) → BN → ReLU → MaxPool(2)
        Conv(32→64, 3) → BN → ReLU → Conv(64→64, 3) → BN → ReLU → MaxPool(2)
        AdaptiveAvgPool(1) → Dropout → FC(64→num_classes)

    Feature dim: 64
    """

    def __init__(self, num_classes=10, dropout_rate=0.0, head_init_scale=1.0):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(64, num_classes)

        if head_init_scale != 1.0:
            with torch.no_grad():
                self.fc.weight.mul_(head_init_scale)
                self.fc.bias.mul_(head_init_scale)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)


def mnist_convnet(num_classes=10, dropout_rate=0.0, head_init_scale=1.0):
    return MNISTConvNet(num_classes=num_classes,
                        dropout_rate=dropout_rate,
                        head_init_scale=head_init_scale)


# ---------------------------------------------------------------------------
# Student model for ensemble distillation
# ---------------------------------------------------------------------------

class MNISTConvNetStudent(nn.Module):
    """Compact CNN student with dual heads: classification logits + scalar EU.

    The EU head receives both the backbone features (64-d) and the detached
    softmax probabilities from the classifier (10-d).
    """

    FEAT_DIM = 64

    def __init__(self, num_classes=10, eu_hidden=128):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.FEAT_DIM, num_classes)

        self.eu_fc1 = nn.Linear(self.FEAT_DIM + num_classes, eu_hidden)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)

        logits = self.fc(feat)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
        eu_in = torch.cat([feat.detach(), probs], dim=-1)
        eu = self.eu_fc2(F.leaky_relu(self.eu_fc1(eu_in), 0.1)).squeeze(-1)
        return logits, eu

    @property
    def eu_head_parameters(self):
        yield from self.eu_fc1.parameters()
        yield from self.eu_fc2.parameters()

    def reinit_eu_head(self):
        nn.init.kaiming_normal_(self.eu_fc1.weight)
        nn.init.zeros_(self.eu_fc1.bias)
        nn.init.kaiming_normal_(self.eu_fc2.weight)
        nn.init.zeros_(self.eu_fc2.bias)


def mnist_convnet_student(num_classes=10, eu_hidden=128):
    return MNISTConvNetStudent(num_classes=num_classes, eu_hidden=eu_hidden)
