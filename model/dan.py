import functools
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


def word_dropout(x_in):
    count_word = x_in.size(1)
    if count_word < 4:
        return x_in
    else:
        i = np.random.randint(count_word)
        return torch.cat([x_in[:, 0:i], x_in[:, i + 1:]], dim=1)


class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = Parameter(torch.FloatTensor(attention_size, 1))
        torch.nn.init.xavier_normal_(self.attention)

    def forward(self, x_in):
        x_in = word_dropout(x_in)
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x  # , attention_score


class DAN(nn.Module):
    def __init__(self, emb_weights):
        super(DAN, self).__init__()
        self.attention = Attention(300)
        self.embedding = nn.Embedding(num_embeddings=50000, embedding_dim=300, _weight=emb_weights)
        self.embedding.weight.requires_grad = False
        self.linear1 = nn.Linear(300, 100)
        self.bn1 = nn.BatchNorm1d(num_features=100)
        self.linear2 = nn.Linear(100, 25)
        self.bn2 = nn.BatchNorm1d(num_features=25)
        self.linear3 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        # x = x.mean(dim=1)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = torch.sigmoid(self.linear3(x))
        return x


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))
