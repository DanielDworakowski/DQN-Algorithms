import torch
import torch.nn as nn

class atari_model(torch.nn.Module):

    def __init__(self, num_actions):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out.view(x.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out