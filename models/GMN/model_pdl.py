import paddle.nn as nn

from models.GMN.affinity_layer_pdl import InnerpAffinity as Affinity
from models.GMN.power_iteration_pdl import PowerIteration
from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from src.lap_solvers_pdl.hungarian import hungarian
from models.GMN.voting_layer_pdl import Voting
from models.GMN.displacement_layer_pdl import Displacement
from src.utils_pdl.build_graphs import reshape_edge_feature
from src.utils_pdl.feature_align import feature_align_fast
from src.utils_pdl.factorize_graph_matching import construct_aff_mat_dense

from src.utils.config import cfg

import src.utils_pdl.backbone
CNN = eval(f'src.utils_pdl.backbone.{cfg.BACKBONE}')


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = Affinity(cfg.GMN.FEATURE_CHANNEL)

        self.power_iteration = PowerIteration(
            max_iter=cfg.GMN.PI_ITER_NUM,
            stop_thresh=cfg.GMN.PI_STOP_THRESH)

        self.bi_stochastic = Sinkhorn(
            max_iter=cfg.GMN.BS_ITER_NUM,
            epsilon=cfg.GMN.BS_EPSILON)

        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)

        self.displacement_layer = Displacement()

        self.l2norm = nn.LocalResponseNorm(
            cfg.GMN.FEATURE_CHANNEL * 2,
            alpha=cfg.GMN.FEATURE_CHANNEL * 2,
            beta=0.5,
            k=0)

        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict, **kwargs):
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            # extract features
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align_fast(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align_fast(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align_fast(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align_fast(tgt_edge, P_tgt, ns_tgt, self.rescale)

        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unsupported data type for this model.')

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)

        # affinity layer
        Me, Mp = self.affinity_layer.forward(X, Y, U_src, U_tgt)

        M = construct_aff_mat_dense(Me, Mp, K_G, K_H)

        v = self.power_iteration.forward(M)
        s = v.reshape([v.shape[0], P_tgt.shape[1], -1]).transpose((0, 2, 1))

        # s = self.voting_layer.forward(s, ns_src, ns_tgt)
        s = self.bi_stochastic.forward(s, ns_src, ns_tgt)

        # d, _ = self.displacement_layer.forward(s, P_src, P_tgt)

        data_dict.update({
            'ds_mat': s,
            'perm_mat': hungarian(s, ns_src, ns_tgt),
            'aff_mat': M
        })
        return data_dict
