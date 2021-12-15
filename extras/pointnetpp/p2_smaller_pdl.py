import paddle
import paddle.nn as nn
from .pointnet_util_pdl import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from pygmtools.dataset import VOC2011_KPT_NAMES


labels = sorted(VOC2011_KPT_NAMES)


class MultiScalePropagation(nn.Layer):
    def __init__(self, additional_channel, g_channel, o_channel):
        super().__init__()
        self.sa = PointNetSetAbstractionMsg(24, [0.1, 0.2, 0.3, 0.6, 1.0], [
                                            24] * 5, 3 + additional_channel, [[64, 128], [128, 256], [64, 128], [32, 64], [24, 48]])
        self.fp = PointNetFeaturePropagation(
            128 + 256 + 128 + 64 + 48 + 3 + additional_channel + g_channel, mlp=[512, 384])
        self.proj = nn.Conv1D(384, o_channel, 1)

    def forward(self, xyz, g):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa(l0_xyz, l0_points)
        g = g.repeat(1, 1, N)
        l0_points = self.fp(l0_xyz, l1_xyz, paddle.concat(
            [g, l0_points], 1), l1_points)
        return self.proj(l0_points)


class get_model(nn.Layer):
    def __init__(self, additional_channel, g_channel, e_channel):
        super(get_model, self).__init__()
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(46, [0.1, 0.2, 0.3, 0.6, 1.0], [
                                             46] * 5, 3 + additional_channel + e_channel, [[96, 128], [192, 256], [96, 128], [56, 64], [40, 64]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 128 + 3, mlp=[400, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(
            in_channel=1024 + 512 + 128, mlp=[1024, 512])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=512 + 6 + 32 * 0 + g_channel + additional_channel, mlp=[512, 256])
        self.conv1 = nn.Conv1D(256, 32, 1)
        # self.cls_emb = nn.Embedding(len(labels), 32)

    def forward(self, xyz, es, g):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, es)
        l2_xyz, l2_points = l1_xyz, l1_points  # self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = l2_points
        # cls_label = self.cls_emb(paddle.tensor([labels.index(i) for i in cls], device=l1_points.device))
        # cls_label_one_hot = cls_label.view(B, 32, 1).repeat(1, 1, N)
        g = g.tile((1, 1, N))
        l0_points = self.fp1(l0_xyz, l1_xyz, paddle.concat(
            [g, l0_xyz, l0_points], 1), l1_points)
        # FC layers
        x = self.conv1(l0_points)
        return l0_points, x
