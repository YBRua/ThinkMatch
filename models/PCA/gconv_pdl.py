import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Gconv(nn.Layer):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """

    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        k = math.sqrt(1.0 / in_features)
        weight_attr_1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_2 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k))

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs,
                              weight_attr=weight_attr_1,
                              bias_attr=bias_attr_1)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs,
                              weight_attr=weight_attr_2,
                              bias_attr=bias_attr_2)

    def forward(self, A, x, norm=True):
        """Forward pass of basic graph convolutional layer

        Args:
            `A` (Tensor): Adjacent matrix for a certain graph
            `x` (Tensor): Node feature
            norm (bool, optional): Whether to apply L1 normalization to adjacency matrix `A`
                Default: True.

        Returns:
            output
        """
        if norm is True:
            A = F.normalize(A, p=1, axis=-2)
        msg_passing = paddle.bmm(A, F.relu(self.a_fc(x)))
        node_passing = F.relu(self.u_fc(x))
        x = msg_passing + node_passing
        return x


class Siamese_Gconv(nn.Layer):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """

    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2
