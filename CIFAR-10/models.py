"""
CIFAR-10 adapted ResNet18 for deep ensembles.

Standard ResNet18 is designed for 224x224 ImageNet images. For CIFAR-10 (32x32),
we replace the initial 7x7 conv (stride 2) + maxpool with a single 3x3 conv
(stride 1, padding 1), keeping the feature map resolution appropriate.

Supports per-member diversity knobs: dropout before classifier, head init scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    """ResNet adapted for CIFAR-10 (32x32 inputs).

    Args:
        dropout_rate: Dropout probability applied before the final FC layer.
            Acts as a regularization diversity knob across ensemble members.
        head_init_scale: Multiplier for the initialization of the classification
            head weights/bias. Different scales bias members toward different
            initial decision boundaries, promoting diversity in early training.
    """

    def __init__(self, block, num_blocks, num_classes=10,
                 dropout_rate=0.0, head_init_scale=1.0):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if head_init_scale != 1.0:
            with torch.no_grad():
                self.fc.weight.mul_(head_init_scale)
                self.fc.bias.mul_(head_init_scale)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)


def cifar_resnet18(num_classes=10, dropout_rate=0.0, head_init_scale=1.0):
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                       dropout_rate=dropout_rate, head_init_scale=head_init_scale)


# ---------------------------------------------------------------------------
# Student model for ensemble distillation
# ---------------------------------------------------------------------------

class CIFARResNetStudent(nn.Module):
    """ResNet18 student with dual heads: classification logits + scalar EU.

    The EU head receives both the backbone features (512-d) and the detached
    softmax probabilities from the classifier (10-d). This gives the EU head
    explicit access to the model's own confidence, which is strongly
    correlated with epistemic uncertainty.
    """

    def __init__(self, block, num_blocks, num_classes=10, eu_hidden=128):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512 * block.expansion
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes)

        # EU head takes feat ⊕ softmax(logits) = feat_dim + num_classes
        self.eu_fc1 = nn.Linear(feat_dim + num_classes, eu_hidden)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)
        self.eu_act = nn.Softplus()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)

        logits = self.fc(feat)
        # Detach so EU loss never backprops into the classifier
        probs = F.softmax(logits, dim=-1).detach()
        eu_in = torch.cat([feat, probs], dim=-1)
        eu = self.eu_act(self.eu_fc2(F.relu(self.eu_fc1(eu_in)))).squeeze(-1)
        return logits, eu

    @property
    def eu_head_parameters(self):
        """Iterator over EU head parameters only (for Phase 2 optimizer)."""
        yield from self.eu_fc1.parameters()
        yield from self.eu_fc2.parameters()

    def reinit_eu_head(self):
        """Re-initialize EU head weights (call before Phase 2)."""
        nn.init.kaiming_normal_(self.eu_fc1.weight)
        nn.init.zeros_(self.eu_fc1.bias)
        nn.init.kaiming_normal_(self.eu_fc2.weight)
        nn.init.zeros_(self.eu_fc2.bias)


def cifar_resnet18_student(num_classes=10, eu_hidden=128):
    return CIFARResNetStudent(BasicBlock, [2, 2, 2, 2],
                              num_classes=num_classes, eu_hidden=eu_hidden)
