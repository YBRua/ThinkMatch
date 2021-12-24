import paddle
import paddle.nn as nn
import math
import numpy as np
from numpy.random import uniform
from paddle import Tensor


class InnerpAffinity(nn.Layer):
    def __init__(self, d):
        """Inner-Product Affinity Layer
        Computes the affinity matrix via inner product from feature space.

        Implementation is based on method proposed in GMN paper.

        `Me = X * Lambda * Y^T`

        `Mp = Ux * Uy^T`

        Args:
            `d`: Scale of weight d

        Parameters:
            `lambda1`
            `lambda2`

            Weight matrix Lambda = [[lambda1, lambda2],
                                    [lambda2, lambda1]]
            where lambda1, lambda2 > 0
        """
        super(InnerpAffinity, self).__init__()
        self.d = d

        # set parameters
        stdv = 1. / math.sqrt(self.d)
        tmp1 = uniform(low=-1*stdv, high=stdv, size=(self.d, self.d))
        tmp2 = uniform(low=-1*stdv, high=stdv, size=(self.d, self.d))
        tmp1 += np.eye(self.d) / 2.0
        tmp2 += np.eye(self.d) / 2.0

        lambda1_attr = paddle.ParamAttr(
            initializer=nn.initializer.Assign(paddle.to_tensor(tmp1, dtype='float64')))
        lambda2_attr = paddle.ParamAttr(
            initializer=nn.initializer.Assign(paddle.to_tensor(tmp2, dtype='float64')))

        self.lambda1 = self.create_parameter(
            [self.d, self.d],
            attr=lambda1_attr)
        self.lambda2 = self.create_parameter(
            [self.d, self.d],
            attr=lambda2_attr)
        self.add_parameter('lambda1', self.lambda1)
        self.add_parameter('lambda2', self.lambda2)

        self.relu = nn.ReLU()

    def forward(
            self,
            X: Tensor,
            Y: Tensor,
            Ux: Tensor,
            Uy: Tensor,
            w1=1,
            w2=1):
        """Forward pass of InnerpAffinity Layer.

        Args:
            `X` (Tensor): Edgewise (i.e. pairwise) feature for graph 1
            `Y` (Tensor): Edgewise (i.e. pairwise) feature for graph 2
            `Ux` (Tensor): Pointwise (i.e. unary) feature for graph 1
            `Uy` (Tensor): Pointwise (i.e. unary) feature for graph 2
            `w1` (int, optional): Weight for lambda1. Defaults to 1.
            `w2` (int, optional): Weight for lambda2. Defaults to 1.

        Returns:
            `Me`: Edgewise affinity matrix
            `Mp`: Pointwise affinity matrix

        NOTE: For more details please refer to the original paper
        """
        assert X.shape[1] == Y.shape[1] == 2 * self.d
        lambda1 = self.relu(
            self.lambda1 + self.lambda1.transpose((1, 0))) * w1
        lambda2 = self.relu(
            self.lambda2 + self.lambda2.transpose((1, 0))) * w2
        weight = paddle.concat(
            (
                paddle.concat((lambda1, lambda2)),
                paddle.concat((lambda2, lambda1))
            ),
            1)
        Me = paddle.matmul(X.transpose((0, 2, 1)), weight)
        Me = paddle.matmul(Me, Y)
        Mp = paddle.matmul(Ux.transpose((0, 2, 1)), Uy)

        return Me, Mp


class GaussianAffinity(nn.Layer):
    def __init__(self, d, sigma):
        """Affinity Layer with Gaussian Kernel.
        Computes the affinity matrix via gaussian kernel from feature space.

        `Me = exp(- L2(X, Y) / sigma)`
        `Mp = Ux * Uy^T`

        Args:
            `d`: Scale of weight d
            `sigma`: Variance for Gaussian kernel

        NOTE: This layer is parameter-free
        """
        super(GaussianAffinity, self).__init__()
        self.d = d
        self.sigma = sigma

    def forward(self, X, Y, Ux=None, Uy=None, ae=1., ap=1.):
        """Forward pass of GaussianAffinity Layer.

        Args:
            `X` (Tensor): Edgewise (i.e. pairwise) feature for graph 1
            `Y` (Tensor): Edgewise (i.e. pairwise) feature for graph 2
            `Ux` (Tensor): Pointwise (i.e. unary) feature for graph 1
            `Uy` (Tensor): Pointwise (i.e. unary) feature for graph 2
            `ae` (float, optional): Weight for `Me`. Defaults to 1.
            `ap` (float, optional): Weight for `Mp`. Defaults to 1.

        Returns:
            `Me`: Edgewise affinity matrix
            `Mp`: Pointwise affinity matrix
        """
        assert X.shape[1] == Y.shape[1] == self.d

        X = X.unsqueeze(-1).expand(*X.shape, Y.shape[2])
        Y = Y.unsqueeze(-2).expand(*Y.shape[:2], X.shape[2], Y.shape[2])
        dist = paddle.sum(paddle.pow(X - Y, 2), axis=1)
        dist[paddle.isnan(dist)] = float("Inf")
        Me = paddle.exp(- dist / self.sigma) * ae

        if Ux is None or Uy is None:
            return Me
        else:
            Mp = paddle.matmul(Ux.transpose((0, 2, 1)), Uy) * ap
            return Me, Mp
