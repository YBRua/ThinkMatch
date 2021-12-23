import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lap_solvers.sinkhorn_new import Sinkhorn


class GNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0, sk_iter=20, sk_tau=0.05, edge_emb=False):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        if edge_emb:
            self.e_func = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat),
                nn.ReLU()
            )
        else:
            self.e_func = None

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            #nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            #nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        if self.e_func is not None:
            W1 = torch.mul(A.unsqueeze(-1), x.unsqueeze(1))
            W2 = torch.cat((W, W1), dim=-1)
            W_new = self.e_func(W2)
        else:
            W_new = W

        if norm is True:
            A = F.normalize(A, p=1, dim=2)

        x1 = self.n_func(x)
        x2 = torch.matmul((A.unsqueeze(-1) * W_new).permute(0, 3, 1, 2),
                          x1.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(-1).transpose(1, 2)
        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0, 2, 1).reshape(
                x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(
                2, 1).contiguous()

            x6 = x5.reshape(x.shape[0], self.sk_channel,
                            n1.max() * n2.max()).permute(0, 2, 1)
            x_new = torch.cat((x2, x6), dim=-1)
        else:
            x_new = x2

        return W_new, x_new
