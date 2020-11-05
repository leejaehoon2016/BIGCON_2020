import torch.nn as nn
import torch

class GRUNet(nn.Module):
    def __init__(self, total_input_size,device):
        super(GRUNet, self).__init__()
        self.dim = total_input_size
        self.second = self.dim * 2
        self.third = self.dim
        self.device = device
        self.first = nn.Sequential(nn.Linear(self.dim, self.second),
                                   nn.BatchNorm1d(self.second),
                                   nn.ReLU())
        self.lstm = nn.GRU(self.second, self.third)
        self.last = nn.Sequential(nn.BatchNorm1d(self.third),
                                  nn.Linear(self.third, self.third // 2),
                                  nn.Tanh(),
                                  nn.Linear(self.third // 2, 1),
                                  nn.Tanh())

    def forward(self, x, zero_pad, broad_id):
        x = self.first(x)
        h = torch.zeros((1, 1, self.third)).to(self.device)
        h_tmp_lst = []
        out_lst = []
        zero_pad_input = torch.zeros_like(x[0].view((1,1,-1)))

        for i in range(len(x)):
            if zero_pad[i] > 0:
                for _ in range(zero_pad[i]):
                    _, h = self.lstm(zero_pad_input, h)
            out, h_tmp = self.lstm(x[i].view((1,1,-1)), h)
            out_lst.append(out)
            if i != len(x) - 1 and broad_id[i] == broad_id[i + 1]:
                h_tmp_lst.append(h_tmp)
            else:
                h_tmp_lst.append(h_tmp)
                h = sum(h_tmp_lst) / len(h_tmp_lst)
                h_tmp_lst = []
        out = torch.cat(out_lst, dim=0).view(x.shape[0],-1)
        out = self.last(out)
        return out

class LSTMNet(nn.Module):
    def __init__(self, total_input_size):
        super(LSTMNet, self).__init__()
        self.dim = total_input_size
        self.first = nn.Sequential(nn.Linear(self.dim, 10),
                                   nn.BatchNorm1d(10))
        self.lstm = nn.LSTM(10, 10)
        self.last = nn.Sequential(nn.BatchNorm1d(10),
                                  nn.Linear(10, 1),
                                  nn.Tanh())

    def forward(self, x, zero_pad, broad_id):
        x = self.first(x)
        h = torch.zeros_like(x[0].view((1, 1, -1)))
        c = torch.zeros_like(x[0].view((1,1,-1)))
        h_tmp_lst = [] ; c_tmp_lst = []
        out_lst = []
        zero_pad_input = torch.zeros_like(x[0].view((1,1,-1)))

        for i in range(len(x)):
            if zero_pad[i] > 0:
                for _ in range(zero_pad[i]):
                    _, (h, c) = self.lstm(zero_pad_input, (h , c))
            out, (h_tmp, c_tmp) = self.lstm(x[i].view((1,1,-1)), (h, c))
            out_lst.append(out)
            if i != len(x) - 1 and broad_id[i] == broad_id[i + 1]:
                h_tmp_lst.append(h_tmp) ; c_tmp_lst.append(c_tmp)
            else:
                h_tmp_lst.append(h_tmp) ; c_tmp_lst.append(c_tmp)
                h = sum(h_tmp_lst) / len(h_tmp_lst) ; c = sum(c_tmp_lst) / len(c_tmp_lst)
                h_tmp_lst = [] ; c_tmp_lst = []
        out = torch.cat(out_lst, dim=0).view(x.shape[0],-1)
        out = self.last(out)
        return out