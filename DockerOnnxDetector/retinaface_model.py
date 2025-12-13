import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp: int, oup: int, stride: int = 1, relu: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_bn_no_relu(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return conv_bn(inp, oup, stride=stride, relu=False)


class DepthWise(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MobileNetV1(nn.Module):
    """MobileNetV1 backbone with width multiplier for the 0.25x RetinaFace variant."""

    def __init__(self, width_mult: float = 0.25):
        super().__init__()
        self.width_mult = width_mult
        scale = lambda c: max(1, int(c * width_mult))

        self.conv1 = conv_bn(3, scale(32), stride=2)

        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (256, 1),  # P3 (stride 8) after this layer
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),  # P4 (stride 16) after this layer
            (1024, 2),
            (1024, 1),  # P5 (stride 32) after this layer
        ]

        layers = []
        in_channels = scale(32)
        for out_channels, stride in cfg:
            oc = scale(out_channels)
            layers.append(DepthWise(in_channels, oc, stride))
            in_channels = oc
        self.stages = nn.ModuleList(layers)
        # Indices in cfg that correspond to stride 8, 16, and 32 respectively.
        self.output_layers = {5, 11, 13}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        outputs = []
        for idx, layer in enumerate(self.stages):
            x = layer(x)
            if idx in self.output_layers:
                outputs.append(x)
        if len(outputs) != 3:
            raise RuntimeError(f"Expected 3 feature maps, got {len(outputs)}")
        return outputs[0], outputs[1], outputs[2]


class FPN(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        self.output1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.output2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)
        self.output3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        self.merge1 = conv_bn(out_channels, out_channels, stride=1)
        self.merge2 = conv_bn(out_channels, out_channels, stride=1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = inputs

        p5 = self.output3(c5)
        p4 = self.output2(c4)
        p3 = self.output1(c3)

        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="nearest")

        p4 = self.merge2(p4)
        p3 = self.merge1(p3)

        return p3, p4, p5


class SSH(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        assert out_channel % 4 == 0, "SSH out_channel should be divisible by 4"
        self.conv3x3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5x5_1 = conv_bn(in_channel, out_channel // 4, stride=1)
        self.conv5x5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7x7_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7 = self.conv7x7_2(conv5x5)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = F.relu(out)
        return out


class ClassHead(nn.Module):
    def __init__(self, in_channel: int, num_anchor: int = 2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channel: int, num_anchor: int = 2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channel: int, num_anchor: int = 2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channel, num_anchor * 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, phase: str = "test", width_mult: float = 0.25):
        super().__init__()
        self.phase = phase
        self.body = MobileNetV1(width_mult=width_mult)

        in_channels = [int(64 * width_mult), int(128 * width_mult), int(256 * width_mult)]
        self.out_channels = 64

        self.fpn = FPN(in_channels, self.out_channels)
        self.ssh1 = SSH(self.out_channels, self.out_channels)
        self.ssh2 = SSH(self.out_channels, self.out_channels)
        self.ssh3 = SSH(self.out_channels, self.out_channels)

        self.ClassHead = self._make_head(ClassHead, num_heads=3)
        self.BboxHead = self._make_head(BboxHead, num_heads=3)
        self.LandmarkHead = self._make_head(LandmarkHead, num_heads=3)

    def _make_head(self, head_cls, num_heads: int) -> nn.ModuleList:
        return nn.ModuleList([head_cls(self.out_channels, num_anchor=2) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = self.body(x)
        fpn_out = self.fpn((c3, c4, c5))
        features = (
            self.ssh1(fpn_out[0]),
            self.ssh2(fpn_out[1]),
            self.ssh3(fpn_out[2]),
        )

        bbox_reg = torch.cat([self.BboxHead[i](features[i]) for i in range(len(features))], dim=1)
        cls = torch.cat([self.ClassHead[i](features[i]) for i in range(len(features))], dim=1)
        ldm_reg = torch.cat([self.LandmarkHead[i](features[i]) for i in range(len(features))], dim=1)

        if self.phase == "test":
            cls = F.softmax(cls, dim=-1)
        return bbox_reg, cls, ldm_reg
