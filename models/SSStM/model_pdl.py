import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import resnet34
from src.utils.config import cfg
from src.lap_solvers_pdl.hungarian import hungarian
from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from extras.pointnetpp import p2_smaller_pdl
# from src.loss_func import PermutationLoss


class ResCls(nn.Layer):
    def __init__(self, n, intro, unit, outro, ndim=1):
        super().__init__()
        BN = [nn.BatchNorm1D, nn.BatchNorm1D,
              nn.BatchNorm2D, nn.BatchNorm3D][ndim]
        CN = [lambda x, y, _: nn.Linear(
            x, y), nn.Conv1D, nn.Conv2D, nn.Conv3D][ndim]
        self.verse = nn.LayerList([BN(unit) for _ in range(n)])
        self.chorus = nn.LayerList([CN(unit, unit, 1) for _ in range(n)])
        self.intro = CN(intro, unit, 1)
        self.outro = CN(unit, outro, 1)

    def forward(self, x):
        x = self.intro(x)
        for chorus, verse in zip(self.chorus, self.verse):
            d = F.relu(verse(x))
            d = chorus(d)
            x = x + d
        return self.outro(x)


def my_align(raw_feature, P, ori_size: tuple):
    return F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)


def batch_features(embeddings, num_vertices):
    res = paddle.concat([
        embedding[:, :num_v]
        for embedding, num_v in zip(embeddings, num_vertices)], axis=-1)
    return res.transpose((1, 0))


def unbatch_features(orig, embeddings, num_vertices):
    res = paddle.zeros_like(orig)
    cum = 0
    for embedding, num_v in zip(res, num_vertices):
        embedding[:, :num_v] = embeddings[cum: cum + num_v].transpose((1, 0))
        cum = cum + num_v
    return res


class Net(nn.Layer):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=False)
        feature_lat = 64 + (64 + 128 + 256 + 512 + 512)
        self.pix2pt_proj = ResCls(1, feature_lat, 512, 256)
        self.pix2cl_proj = ResCls(1, 1024, 512, 128)
        self.edge_gate = ResCls(1, feature_lat * 3, 512, 1)
        self.edge_proj = ResCls(1, feature_lat * 3, 512, 64)
        self.tau = cfg.IGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.pn = p2_smaller_pdl.get_model(256, 128, 64)
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.IGM.SK_ITER_NUM,
            tau=self.tau,
            epsilon=cfg.IGM.SK_EPSILON
        )
        self.backbone_params = list(self.resnet.parameters())

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        yield x
        x = r.maxpool(x)

        x = r.layer1(x)
        yield x
        x = r.layer2(x)
        yield x
        x = r.layer3(x)
        yield x
        x = r.layer4(x)
        yield x
        x = r.avgpool(x)
        yield x

    def halo(self, feat_srcs, feat_tgts, P_src, P_tgt):
        U_src = paddle.concat([
            my_align(feat_src, P_src, self.rescale) for feat_src in feat_srcs
        ], axis=1)
        U_tgt = paddle.concat([
            my_align(feat_tgt, P_tgt, self.rescale) for feat_tgt in feat_tgts
        ], axis=1)
        glob_src = feat_srcs[-1].flatten(1).unsqueeze(-1)
        glob_tgt = feat_tgts[-1].flatten(1).unsqueeze(-1)
        # F_src = torch.cat([
        #     U_src,
        #     glob_tgt.expand(*glob_tgt.shape[:-1], U_src.shape[-1])
        # ], 1)
        # F_tgt = torch.cat([
        #     U_tgt,
        #     glob_src.expand(*glob_src.shape[:-1], U_tgt.shape[-1])
        # ], 1)
        ghalo_src = paddle.concat((glob_src, glob_tgt), axis=1)
        ghalo_tgt = paddle.concat((glob_tgt, glob_src), axis=1)
        return U_src, U_tgt, ghalo_src, ghalo_tgt

    def edge_activations(self, feats, F_, P, n):
        # F: BCN
        # P: BN2
        # n: B
        ep = ((P.unsqueeze(-2) + P.unsqueeze(-3)) / 2).flatten(1, 2)  # B N^2 2
        L = (
            paddle.concat([F_, paddle.zeros_like(F_)], 1).unsqueeze(-1) +
            paddle.concat([paddle.zeros_like(F_), F_], 1).unsqueeze(-2)
        ).flatten(2)  # B2CN^2
        E = paddle.concat([
            my_align(feat, ep, self.rescale) for feat in feats
        ], axis=1)  # BCN^2
        CE = paddle.concat([L, E], axis=1)
        mask = paddle.arange(
            F_.shape[-1]).expand((len(F_), F_.shape[-1])) < n.unsqueeze(-1)
        # BN
        mask = paddle.logical_and(mask.unsqueeze(-2), mask.unsqueeze(-1))
        mask = paddle.cast(mask, 'float32')
        return (
            F.sigmoid(self.edge_gate(CE))
            * F.normalize(self.edge_proj(CE), axis=1)
            * mask.flatten(1).unsqueeze(1))\
            .reshape((F_.shape[0], -1, F_.shape[-1], F_.shape[-1]))

    def points(
            self,
            y_src, y_tgt,
            P_src, P_tgt,
            n_src, n_tgt,
            e_src, e_tgt, g):
        resc = paddle.to_tensor(self.rescale)
        P_src, P_tgt = P_src / resc, P_tgt / resc
        P_src, P_tgt = P_src.transpose((0, 2, 1)), P_tgt.transpose((0, 2, 1))
        # not used during inference
        # if self.training:
        #     P_src = P_src + torch.rand_like(P_src)[..., :1] * 0.2 - 0.1
        #     P_tgt = P_tgt + torch.rand_like(P_tgt)[..., :1] * 0.2 - 0.1
        key_mask_src = paddle.arange(y_src.shape[-1])\
            .expand((len(y_src), y_src.shape[-1])) < n_src.unsqueeze(-1)
        key_mask_tgt = paddle.arange(y_tgt.shape[-1])\
            .expand((len(y_tgt), y_tgt.shape[-1])) < n_tgt.unsqueeze(-1)
        key_mask_cat = paddle.concat(
            (key_mask_src, key_mask_tgt), -1).unsqueeze(1)
        P_src = paddle.concat((P_src, paddle.zeros_like(P_src[:, :1])), 1)
        P_tgt = paddle.concat((P_tgt, paddle.ones_like(P_tgt[:, :1])), 1)
        pcd = paddle.concat((P_src, P_tgt), -1)
        y_cat = paddle.concat((y_src, y_tgt), -1)
        e_cat = paddle.zeros([
            e_src.shape[0], e_src.shape[1],
            e_src.shape[2] + e_tgt.shape[2], e_src.shape[3] + e_tgt.shape[3]
        ], dtype=e_src.dtype)
        e_cat[..., :e_src.shape[2], :e_src.shape[3]] = e_src
        e_cat[..., e_src.shape[2]:, e_src.shape[3]:] = e_tgt
        r1, r2 = self.pn(
            paddle.concat((pcd, y_cat), 1) * key_mask_cat,
            e_cat, g)
        return r1[:, :, :y_src.shape[-1]], r2[:, :, :y_src.shape[-1]]

    def forward(self, data_dict, **kwargs):
        src, tgt = data_dict['images']
        P_src, P_tgt = data_dict['Ps']
        ns_src, ns_tgt = data_dict['ns']

        feat_srcs, feat_tgts = [], []
        for feat in self.encode(paddle.concat([src, tgt])):
            feat_srcs.append(feat[:len(src)])
            feat_tgts.append(feat[len(src):])
        if self.training:
            P_src = P_src + paddle.randn(P_src.shape, dtype='float32') * 2 - 1
            P_tgt = P_tgt + paddle.randn(P_tgt.shape, dtype='float32') * 2 - 1
        F_src, F_tgt, g_src, g_tgt = self.halo(
            feat_srcs, feat_tgts, P_src, P_tgt)

        ea_src = self.edge_activations(feat_srcs, F_src, P_src, ns_src)
        ea_tgt = self.edge_activations(feat_tgts, F_tgt, P_tgt, ns_tgt)

        y_src, y_tgt = self.pix2pt_proj(F_src), self.pix2pt_proj(F_tgt)

        g_src, g_tgt = self.pix2cl_proj(g_src), self.pix2cl_proj(g_tgt)
        y_src, y_tgt = F.normalize(y_src, axis=1), F.normalize(y_tgt, axis=1)
        g_src, g_tgt = F.normalize(g_src, axis=1), F.normalize(g_tgt, axis=1)

        ff_src, folding_src = self.points(
            y_src, y_tgt, P_src, P_tgt, ns_src, ns_tgt, ea_src, ea_tgt, g_src)
        ff_tgt, folding_tgt = self.points(
            y_tgt, y_src, P_tgt, P_src, ns_tgt, ns_src, ea_tgt, ea_src, g_tgt)

        # sim = paddle.einsum(
        #     'bci,bcj->bij',
        #     folding_src,
        #     folding_tgt
        # )
        # workaround for paddle.einsum (added in 2.2, but we are in 2.1)
        sim = paddle.bmm(folding_src.transpose((0, 2, 1)), folding_tgt)
        data_dict['ds_mat'] = self.sinkhorn(
            sim, ns_src, ns_tgt, dummy_row=True)
        data_dict['perm_mat'] = hungarian(data_dict['ds_mat'], ns_src, ns_tgt)
        data_dict['ff'] = [ff_src, ff_tgt]
        data_dict['gf'] = [g_src, g_tgt]
        return data_dict
