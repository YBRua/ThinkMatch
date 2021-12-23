import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from src.lap_solvers_pdl.sinkhorn import Sinkhorn


class GNNLayer(nn.Layer):
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
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
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
            W1 = paddle.mul(A.unsqueeze(-1), x.unsqueeze(1))
            W2 = paddle.concat((W, W1), axis=-1)
            W_new = self.e_func(W2)
        else:
            W_new = W

        if norm is True:
            A = F.normalize(A, p=1, axis=2)

        x1 = self.n_func(x)
        x2 = paddle.matmul((A.unsqueeze(-1) * W_new).transpose((0, 3, 1, 2)),
                           x1.unsqueeze(2).transpose((0, 3, 1, 2))).squeeze(-1).transpose((0, 2, 1))
        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            # np workaround for repeat_interleave.
            # FIXME: Does NOT support backward propagation
            # see <https://github.com/PaddlePaddle/Paddle/issues/37227>
            n1_rep = paddle.to_tensor(np.repeat(n1, self.sk_channel, axis=0))
            n2_rep = paddle.to_tensor(np.repeat(n2, self.sk_channel, axis=0))
            # n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            # n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.transpose((0, 2, 1)).reshape(
                x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose((0, 2, 1))
            x5 = self.sk(x4, n1_rep, n2_rep,
                         dummy_row=True).transpose((0, 2, 1))

            x6 = x5.reshape(x.shape[0], self.sk_channel,
                            n1.max() * n2.max()).transpose((0, 2, 1))
            x_new = paddle.concat((x2, x6), axis=-1)
        else:
            x_new = x2

        return W_new, x_new
