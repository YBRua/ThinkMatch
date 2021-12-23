import paddle
from paddle import Tensor
from paddle.autograd import PyLayer
from src.utils.sparse import bilinear_diag_torch
import numpy as np


def construct_aff_mat_dense(Ke: Tensor, Kp: Tensor, KroG: Tensor, KroH: Tensor,
                      KroGt: Tensor=None, KroHt: Tensor=None) -> Tensor:
    r"""
    Construct the complete affinity matrix with edge-wise affinity matrix :math:`\mathbf{K}_e`, node-wise matrix
    :math:`\mathbf{K}_p` and graph connectivity matrices :math:`\mathbf{G}_1, \mathbf{H}_1, \mathbf{G}_2, \mathbf{H}_2`

    .. math ::
        \mathbf{K}=\mathrm{diag}(\mathrm{vec}(\mathbf{K}_p)) +
        (\mathbf{G}_2 \otimes_{\mathcal{K}} \mathbf{G}_1) \mathrm{diag}(\mathrm{vec}(\mathbf{K}_e))
        (\mathbf{H}_2 \otimes_{\mathcal{K}} \mathbf{H}_1)^\top

    where :math:`\mathrm{diag}(\cdot)` means building a diagonal matrix based on the given vector,
    and :math:`\mathrm{vec}(\cdot)` means column-wise vectorization.
    :math:`\otimes_{\mathcal{K}}` denotes Kronecker product.

    This function supports batched operations. This formulation is developed by `"F. Zhou and F. Torre. Factorized
    Graph Matching. TPAMI 2015." <http://www.f-zhou.com/gm/2015_PAMI_FGM_Draft.pdf>`_

    :param Ke: :math:`(b\times p\times q)` edge-wise affinity matrix.
     :math:`p`: number of edges in graph 1, :math:`q`: number of edges in graph 2
    :param Kp: :math:`(b\times n\times m)` node-wise affinity matrix.
     :math:`n`: number of nodes in graph 1, :math:`m`: number of nodes in graph 2
    :param KroG: :math:`(b\times nm \times pq)` kronecker product of
     :math:`\mathbf{G}_2 (b\times m \times q)`, :math:`\mathbf{G}_1 (b\times n \times p)`
    :param KroH: :math:`(b\times nm \times pq)` kronecker product of
     :math:`\mathbf{H}_2 (b\times m \times q)`, :math:`\mathbf{H}_1 (b\times n \times p)`
    :param KroGt: transpose of KroG (should be CSR, optional)
    :param KroHt: transpose of KroH (should be CSC, optional)
    :return: affinity matrix :math:`\mathbf K`

    .. note ::
        This function is optimized with sparse CSR and CSC matrices with GPU support for both forward and backward
        computation with PyTorch. To use this function, you need to install ``ninja-build``, ``gcc-7``, ``nvcc`` (which
        comes along with CUDA development tools) to successfully compile our customized CUDA code for CSR and CSC
        matrices. The compiler is automatically called upon requirement.

    For a graph matching problem with 5 nodes and 4 nodes,
    the connection of :math:`\mathbf K` and :math:`\mathbf{K}_p, \mathbf{K}_e` is illustrated as

    .. image :: ../../images/factorized_graph_matching.png

    where :math:`\mathbf K (20 \times 20)` is the complete affinity matrix, :math:`\mathbf{K}_p (5 \times 4)` is the
    node-wise affinity matrix, :math:`\mathbf{K}_e(16 \times 10)` is the edge-wise affinity matrix.
    """
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    Ke_diag = paddle.zeros((B, P * Q, P * Q))
    for b in range(B):
        Ke_diag[b] = paddle.diag(Ke[b].reshape(-1))
    KroG_diag = paddle.bmm(KroG, Ke_diag)  # B, NM, PQ
    KroG_diag_KroH = paddle.bmm(KroG_diag, KroH.transpose((0, 2, 1)))  # B, MN, MN
    res = paddle.zeros((B, M * N, M * N))
    for b in range(B):
        res[b] = paddle.diag(Kp[b].reshape(-1)) + KroG_diag_KroH[b]

    return res
