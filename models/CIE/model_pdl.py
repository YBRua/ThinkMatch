import paddle
import paddle.nn as nn

from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from src.utils_pdl.feature_align import feature_align
from src.utils_pdl.gconv import Siamese_ChannelIndependentConv
from models.PCA.affinity_layer_pdl import Affinity
from src.lap_solvers_pdl.hungarian import hungarian

from src.utils.config import cfg

import src.utils_pdl.backbone
CNN = eval(f'src.utils_pdl.backbone.{cfg.BACKBONE}')


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.CIE.SK_ITER_NUM,
            epsilon=cfg.CIE.SK_EPSILON,
            tau=cfg.CIE.SK_TAU)
        self.l2norm = nn.LocalResponseNorm(
            cfg.CIE.FEATURE_CHANNEL * 2,
            alpha=cfg.CIE.FEATURE_CHANNEL * 2,
            beta=0.5, k=0)

        # currently only the default architecture
        # with 2 GNN layers is supported
        # self.gnn_layer_0 = Siamese_ChannelIndependentConv(
        #     cfg.CIE.FEATURE_CHANNEL * 2, cfg.CIE.GNN_FEAT, 1)
        # self.gnn_layer_1 = Siamese_ChannelIndependentConv(
        #     cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT)

        # self.affinity_0 = Affinity(cfg.CIE.GNN_FEAT)
        # self.affinity_1 = Affinity(cfg.CIE.GNN_FEAT)

        # self.cross_graph_0 = nn.Linear(
        #     cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT)
        # self.cross_graph_edge_0 = nn.Linear(
        #     cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT)

        # TODO: Add support for variable number of GNN layers
        self.gnn_layer = cfg.CIE.GNN_LAYER  # numbur of GNN layers
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_ChannelIndependentConv(
                    cfg.CIE.FEATURE_CHANNEL * 2, cfg.CIE.GNN_FEAT, 1)
            else:
                gnn_layer = Siamese_ChannelIndependentConv(
                    cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(
                i), Affinity(cfg.CIE.GNN_FEAT))
            # only second last layer will have cross-graph module
            if i == self.gnn_layer - 2:
                self.add_module('cross_graph_{}'.format(i), nn.Linear(
                    cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT))
                self.add_module('cross_graph_edge_{}'.format(
                    i), nn.Linear(cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT))

        self.rescale = cfg.PROBLEM.RESCALE

    def add_module(self, key, module):
        setattr(self, key, module)

    def forward(self, data_dict, **kwargs):
        if 'images' in data_dict:
            # real image data
            src, tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['images']]
            P_src, P_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['Ps']]
            ns_src, ns_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['ns']]
            G_src, G_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['Gs']]
            H_src, H_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['Hs']]
            # extract feature
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
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['features']]
            ns_src, ns_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['ns']]
            G_src, G_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['Gs']]
            H_src, H_tgt = [paddle.to_tensor(data=_, dtype='float32') for _ in data_dict['Hs']]

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        P_src_dis = (P_src.unsqueeze(1) - P_src.unsqueeze(2))
        P_src_dis = paddle.norm(P_src_dis, p=2, axis=3).detach()
        P_tgt_dis = (P_tgt.unsqueeze(1) - P_tgt.unsqueeze(2))
        P_tgt_dis = paddle.norm(P_tgt_dis, p=2, axis=3).detach()

        Q_src = paddle.exp(-P_src_dis / self.rescale[0])
        Q_tgt = paddle.exp(-P_tgt_dis / self.rescale[0])

        emb_edge1 = Q_src.unsqueeze(-1)
        emb_edge2 = Q_tgt.unsqueeze(-1)

        # adjacency matrices
        A_src = paddle.bmm(G_src, paddle.transpose(H_src, (0, 2, 1)))
        A_tgt = paddle.bmm(G_tgt, paddle.transpose(H_tgt, (0, 2, 1)))

        # U_src, F_src are features at different scales
        emb1 = paddle.transpose(
            paddle.concat((U_src, F_src), axis=1), (0, 2, 1))
        emb2 = paddle.transpose(
            paddle.concat((U_tgt, F_tgt), axis=1), (0, 2, 1))
        ss = []

        # emb1, emb2, emb_edge1, emb_edge2 = self.gnn_layer_0(
        #     [A_src, emb1, emb_edge1], [A_tgt, emb2, emb_edge2])

        # s = self.affinity_0(emb1, emb2)
        # s = self.sinkhorn(s, ns_src, ns_tgt)
        # ss.append(s)

        # new_emb1 = self.cross_graph_0(
        #     paddle.concat((emb1, paddle.bmm(s, emb2)), dim=-1))
        # new_emb2 = self.cross_graph_0(
        #     paddle.concat(
        #         (emb2, paddle.bmm(s.transpose(1, 2), emb1)), dim=-1))

        # emb1 = new_emb1
        # emb2 = new_emb2

        # emb1, emb2, emb_edge1, emb_edge2 = self.gnn_layer_1(
        #     [A_src, emb1, emb_edge1], [A_tgt, emb2, emb_edge2])

        # s = self.affinity_1(emb1, emb2)
        # s = self.sinkhorn(s, ns_src, ns_tgt)
        # ss.append(s)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))

            # during forward process, the network structure will not change
            emb1, emb2, emb_edge1, emb_edge2 = gnn_layer(
                [A_src, emb1, emb_edge1], [A_tgt, emb2, emb_edge2])

            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)  # xAx^T

            s = self.sinkhorn(s, ns_src, ns_tgt)
            ss.append(s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                new_emb1 = cross_graph(
                    paddle.concat(
                        (emb1, paddle.bmm(s, emb2)), axis=-1))
                new_emb2 = cross_graph(
                    paddle.concat(
                        (emb2, paddle.bmm(s.transpose((0, 2, 1)), emb1)),
                        axis=-1))
                emb1 = new_emb1
                emb2 = new_emb2

        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        return data_dict
