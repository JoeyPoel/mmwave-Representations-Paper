"""
Unified PyTorch Model Architectures for mmWave Radar Fall Detection.

Includes:
- ResNet18Adaptor: Adapted ResNet-18 for 2D Doppler-Time Map (Rep 1) & 2D Orthogonal Projections (Rep 2)
- PointNetPlusPlusFallDetector: PointNet++ spatial feature extractor with 1D Temporal CNN for 3D Point Sets (Rep 3)
- Radar4DCNN: 2D Spatial-Temporal CNN baseline

All models are designed to be clean, readable, and easily editable.
To tweak network capacity or hyperparameters, simply edit the layer dimensions or dropout rates marked below.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ResNet-18 Adaptor (Representation 1 & Representation 2)
# ==============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        # 3x3 Convolution 1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 Convolution 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Residual shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Adaptor(nn.Module):
    """
    ResNet-18 architecture adapted for 2D radar grids (32x32).
    Input: (B, 3, 32, 32) -> Output: (B, 2) logits for [ADL, Fall].
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(ResNet18Adaptor, self).__init__()
        
        # EDIT HERE: Change base channels (default: 32) to make the model wider or narrower
        self.in_planes = 32

        # Initial stem convolution
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # ResNet residual block layers
        self.layer1 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)

        # Global Average Pooling & Fully-Connected Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # EDIT HERE: Adjust dropout probability (default: 0.3) or classifier hidden size (default: 64)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# Aliases for Representation 1 & Representation 2
ResNet18Spectrogram = ResNet18Adaptor
ResNet18Projections = ResNet18Adaptor


# ==============================================================================
# 2. PointNet++ Fall Detector (Representation 3)
# ==============================================================================

class PointNetPlusPlusFallDetector(nn.Module):
    """
    PointNet++ for 3D Point Set Sequences.
    Processes 10 temporal frames with 32 points per frame.
    Input shape: (B, 5, 320) -> Reshaped per-frame to (B*10, 5, 32).
    Uses dual Max + Avg spatial pooling per frame and 1D Temporal CNN.
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 2):
        super(PointNetPlusPlusFallDetector, self).__init__()
        
        self.sa1_conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.sa1_conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.sa2_conv1 = nn.Conv1d(128, 256, kernel_size=1)
        self.sa2_conv2 = nn.Conv1d(256, 512, kernel_size=1)
        
        self.bn_sa1 = nn.BatchNorm1d(128)
        self.bn_sa2 = nn.BatchNorm1d(512)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )

        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_frames = x.view(B * 10, 5, 32)

        f1 = F.relu(self.sa1_conv1(x_frames))
        f1 = F.relu(self.bn_sa1(self.sa1_conv2(f1)))

        f2 = F.relu(self.sa2_conv1(f1))
        f2 = F.relu(self.bn_sa2(self.sa2_conv2(f2)))

        # Dual Pooling: Max + Avg
        f_max = torch.max(f2, dim=2)[0]
        f_avg = torch.mean(f2, dim=2)
        frame_feat = (f_max + f_avg) / 2.0

        temp_seq = frame_feat.view(B, 10, 512).permute(0, 2, 1)
        temp_feat = self.temporal_conv(temp_seq)

        p_max = self.pool_max(temp_feat).squeeze(-1)
        p_avg = self.pool_avg(temp_feat).squeeze(-1)
        combined = torch.cat([p_max, p_avg], dim=1)

        return self.fc(combined)

PointNetFallDetector = PointNetPlusPlusFallDetector



# ==============================================================================
# 3. Radar 4D-CNN Baseline
# ==============================================================================

class Radar4DCNN(nn.Module):
    """2D Spatial-Temporal CNN for 4D mmWave Radar Point Clouds."""
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super(Radar4DCNN, self).__init__()
        
        # EDIT HERE: Adjust filter channels or pooling kernel sizes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ==============================================================================
# Helper Function: Model Summary & Parameter Audit
# ==============================================================================

def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> dict:
    """Returns parameter count, trainable parameter count, and tensor shapes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_in = torch.zeros((1,) + input_shape)
    model.eval()
    with torch.no_grad():
        out = model(dummy_in)
    return {
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_shape": list(dummy_in.shape),
        "output_shape": list(out.shape)
    }


