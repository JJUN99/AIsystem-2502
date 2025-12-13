import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet, self).__init__()
        # Load pretrained MobileNetV2
        self.pretrained_net = mobilenet_v2(pretrained=True)
        self.features = self.pretrained_net.features
        
        # Adding the extra layers
        self.conv2d = nn.Conv2d(1280, 32, kernel_size=(3, 3), padding=1)  # Adjust input channels if needed
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x