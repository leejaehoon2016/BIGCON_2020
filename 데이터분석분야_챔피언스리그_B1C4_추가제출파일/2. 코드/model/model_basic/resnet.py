import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self,total_input_size):
        super(ResNet, self).__init__()
        dim = total_input_size
        self.layer1 = nn.Sequential(nn.Linear(dim, dim * 2),
                                    nn.BatchNorm1d(dim * 2),
                                    nn.CELU())
        self.res1 = nn.Sequential(nn.Linear(dim * 2, dim * 2),
                                  nn.BatchNorm1d(dim * 2),
                                  nn.CELU())
        self.layer2 = nn.Sequential(nn.Linear(dim * 2, dim * 2),
                                    nn.BatchNorm1d(dim * 2),
                                    nn.CELU())
        self.res2 = nn.Sequential(nn.Linear(dim * 2, dim * 2),
                                  nn.BatchNorm1d(dim * 2),
                                  nn.CELU())
        self.layer3 = nn.Sequential(nn.Linear(dim * 2, dim),
                                    nn.BatchNorm1d(dim),
                                    nn.Linear(dim, 1))

        self.last = nn.Tanh()


    def forward(self, x):
        x = self.layer1(x)
        x = x + self.res1(x)
        x = self.layer2(x)
        x = x + self.res2(x)
        x = self.layer3(x)
        out = self.last(x)
        return out