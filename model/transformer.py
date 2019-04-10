import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class MultiHeadAttention(nn.Module):
    """
        A multihead attention module,
        using scaled dot-product attention.
    """

    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_size = int(self.hidden_size / num_heads)

        self.q_linear = nn.Linear(self.input_size, self.hidden_size)
        self.k_linear = nn.Linear(self.input_size, self.hidden_size)
        self.v_linear = nn.Linear(self.input_size, self.hidden_size)

        self.joint_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # project the queries, keys and values by their respective weight matrices
        q_proj = self.q_linear(q).view(q.size(0), q.size(1), self.num_heads, self.head_size).transpose(1, 2)
        k_proj = self.k_linear(k).view(k.size(0), k.size(1), self.num_heads, self.head_size).transpose(1, 2)
        v_proj = self.v_linear(v).view(v.size(0), v.size(1), self.num_heads, self.head_size).transpose(1, 2)

        # calculate attention weights
        unscaled_weights = torch.matmul(q_proj, k_proj.transpose(2, 3))
        weights = self.softmax(unscaled_weights / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(unscaled_weights)))

        # weight values by their corresponding attention weights
        weighted_v = torch.matmul(weights, v_proj)
        weighted_v = weighted_v.transpose(1, 2).contiguous()

        # do a linear projection of the weighted sums of values
        joint_proj = self.joint_linear(weighted_v.view(q.size(0), q.size(1), self.hidden_size))

        # store a reference to attention weights, for THIS forward pass,
        # for visualisation purposes
        self.weights = weights

        return joint_proj


class Block(nn.Module):
    """
        One block of the transformer.
        Contains a multihead attention sublayer
        followed by a feed forward network.
    """

    def __init__(self, input_size, hidden_size, num_heads, activation=nn.ReLU, dropout=None):
        super(Block, self).__init__()
        self.dropout = dropout

        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(input_size)

        ff_layers = [
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, input_size),
        ]

        if self.dropout:
            self.attention_dropout = nn.Dropout(dropout)
            ff_layers.append(nn.Dropout(dropout))

        self.ff = nn.Sequential(
            *ff_layers
        )
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        attended = self.attention_norm(self.attention_dropout(self.attention(x, x, x)) + x)
        return self.ff_norm(self.ff(attended) + x)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_blocks, num_heads, activation=nn.ReLU, dropout=None):
        """
            A single Transformer Network
        """
        super(Transformer, self).__init__()

        self.blocks = nn.Sequential(
            *[Block(input_size, hidden_size, num_heads, activation, dropout=dropout)
              for _ in np.arange(num_blocks)]
        )

    def forward(self, x):
        """
            Sequentially applies the blocks of the Transformer network
        """
        return self.blocks(x)


class Net(nn.Module):
    """
        A neural network that encodes a sequence
        using a Transformer network
    """

    def __init__(self, embeddings, max_length=config.max_seq_length, model_size=128, num_heads=4, num_blocks=1, dropout=0.1,
                 freeze_embeddings=False):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
        self.model_size = model_size
        self.emb_ff = nn.Linear(embeddings.size(1), self.model_size)
        self.pos = nn.Linear(max_length, self.model_size)
        self.max_length = max_length
        self.transformer = Transformer(
            self.model_size,
            self.model_size,
            self.model_size,
            num_blocks,
            num_heads,
            dropout=dropout
        )
        self.linear1 = nn.Linear(self.model_size, int(self.model_size/2))
        self.bn = nn.BatchNorm1d(int(self.model_size/2))
        self.linear2 = nn.Linear(int(self.model_size/2), 1)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1)
        x = self.emb_ff(self.embeddings(x))
        pos = self.pos(self.get_pos_onehot().to(x)).unsqueeze(0)
        x = x.view(*(x_size + (self.model_size,)))
        x += pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = F.elu(self.linear1(x))
        x = self.bn(x)
        x = torch.sigmoid(self.linear2(x))
        return x

    def get_pos_onehot(self):
        onehot = torch.zeros(self.max_length, self.max_length)
        idxs = torch.arange(self.max_length).long().view(-1, 1)
        onehot.scatter_(1, idxs, 1)
        return onehot
