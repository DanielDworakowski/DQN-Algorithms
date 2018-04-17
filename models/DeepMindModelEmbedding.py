import torch
import torch.nn as nn

class atari_model(torch.nn.Module):

    def __init__(self, num_actions):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.convTrans1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.convTrans2 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.convTrans3 = nn.ConvTranspose2d(32, 4, 8, stride=4)

        self.x = None
        self.reconst = None

    def forward(self, x):
        self.x = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)

        outAction = out.view(x.shape[0], -1)
        outAction = self.fc1(outAction)
        outAction = self.relu(outAction)
        outAction = self.fc2(outAction)

        outReconst = self.convTrans1(out)
        outReconst = self.relu(outReconst)
        outReconst = self.convTrans2(outReconst)
        outReconst = self.relu(outReconst)
        outReconst = self.convTrans3(outReconst)
        outReconst = torch.nn.functional.tanh(outReconst)
        self.reconst = outReconst

        return outAction