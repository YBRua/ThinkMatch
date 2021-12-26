import paddle
import paddle.nn as nn

from src.lap_solvers_pdl.sinkhorn import Sinkhorn
from src.utils_pdl.feature_align import feature_align_fast
from models.PCA.gconv_pdl import Siamese_Gconv
from models.PCA.affinity_layer_pdl import Affinity
from src.lap_solvers_pdl.hungarian import hungarian

from src.utils.config import cfg

import src.utils_pdl.backbone
CNN = eval(f'src.utils_pdl.backbone.{cfg.BACKBONE}')


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(
            max_iter=cfg.PCA.SK_ITER_NUM,
            epsilon=cfg.PCA.SK_EPSILON,
            tau=cfg.PCA.SK_TAU)
        self.l2norm = nn.LocalResponseNorm(
            cfg.PCA.FEATURE_CHANNEL * 2,
            alpha=cfg.PCA.FEATURE_CHANNEL * 2,
            beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(
                    cfg.PCA.FEATURE_CHANNEL * 2,
                    cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(
                    cfg.PCA.GNN_FEAT,
                    cfg.PCA.GNN_FEAT)
            self.add_module(
                f'gnn_layer_{i}', gnn_layer)
            self.add_module(
                f'affinity_{i}', Affinity(cfg.PCA.GNN_FEAT))
            # only second last layer will have cross-graph module
            if i == self.gnn_layer - 2:
                self.add_module(
                    f'cross_graph_{i}',
                    nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))
        self.cross_iter = cfg.PCA.CROSS_ITER
        self.cross_iter_num = cfg.PCA.CROSS_ITER_NUM
        self.rescale = cfg.PROBLEM.RESCALE

    def add_module(self, key, module):
        setattr(self, key, module)

    def reload_backbone(self):
        self.node_layers, self.edge_layers = self.get_backbone(True)

    def forward(self, data_dict, **kwargs):
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

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
            U_src = feature_align_fast(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align_fast(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align_fast(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align_fast(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        emb1 = paddle.concat((U_src, F_src), axis=1).transpose((0, 2, 1))
        emb2 = paddle.concat((U_tgt, F_tgt), axis=1).transpose((0, 2, 1))
        ss = []

        if not self.cross_iter:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, f'gnn_layer_{i}')
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, f'affinity_{i}')
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                ss.append(s)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, f'cross_graph_{i}')
                    new_emb1 = cross_graph(
                        paddle.concat(
                            (emb1, paddle.bmm(s, emb2)),
                            axis=-1))
                    new_emb2 = cross_graph(
                        paddle.concat(
                            (emb2, paddle.bmm(s.transpose((0, 2, 1)), emb1)),
                            axis=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, f'gnn_layer_{i}')
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = paddle.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]))

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, f'cross_graph_{i}')
                emb1 = cross_graph(
                    paddle.concat(
                        (emb1_0, paddle.bmm(s, emb2_0)),
                        axis=-1))
                emb2 = cross_graph(
                    paddle.concat(
                        (emb2_0, paddle.bmm(s.transpose((0, 2, 1)), emb1_0)),
                        axis=-1))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, f'gnn_layer_{i}')
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, f'affinity_{i}')
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)

        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        return data_dict
