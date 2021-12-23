import paddle
from typing import List
from paddle import Tensor
from paddle.autograd import PyLayer
from src.utils.sparse import bilinear_diag_torch
import numpy as np


def construct_aff_mat_dense_slower(Ke: Tensor, Kp: Tensor, KroG: List[Tensor], KroH: List[Tensor]) -> Tensor:
    """Construct the complete affinity matrix with
    edge-wise affinity matrix Ke, node-wise matrix Kp
    and Kronecker products KroG, KroH.

    M = diag(vec(Kp)) + KroG @ diag(vec(Ke)) @ KroH.T

    where diag(x) constructs a diagnal matrix with vector x on the main diag
    and vec(x) applies COLUMN vectorization to matrix x

    Args:
        Ke (Tensor): (B, P, Q) edge-wise affinity matrix.
            P: number of edges in graph 1, Q: number of edges in graph 2
        Kp (Tensor): (B, M, N) node-wise affinity matrix.
            N: number of nodes in graph 1, M: number of nodes in graph 2
        KroG (List[Tensor]): A list of length B,
            containing (MN, PQ) Kronecker product of G2 and G1
            G1: (B, N, P). G2: (B, M, Q)
        KroH (List[Tensor]): A list of length B,
            containing (MN, PQ) Kronecker product of H2 and H1
            H1: (B, N, P). H2: (B, M, Q)

    Returns:
        Tensor: Affinity Matrix K

    NOTE: This implementation is a workaround for the original
    CSR/CSC sparse implementation with GPU support in the torch version.
    Since Paddle currently does not have adequate support for C++ extensions
    It is dense and devecotrized, and can thus its performance is suboptimal.
    """
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    # NOTE: Every single transpose here matters
    # because we should use a COLUMN VECTORIZATION here.
    res = paddle.zeros((B, M * N, M * N))
    for b in range(B):
        KroG_diag = paddle.matmul(
            KroG[b],
            paddle.diag(Ke[b].transpose((1, 0)).reshape([-1])))  # MN, PQ
        KroG_diag_KroH = paddle.matmul(
            KroG_diag,
            KroH[b].transpose((1, 0)))  # MN, MN
        res[b] = paddle.diag(
            Kp[b].transpose((1, 0)).reshape([-1])) + KroG_diag_KroH

    return res


def construct_aff_mat_dense(Ke: Tensor, Kp: Tensor, KroG: List[Tensor], KroH: List[Tensor]) -> Tensor:
    """Construct the complete affinity matrix with
    edge-wise affinity matrix Ke, node-wise matrix Kp
    and Kronecker products KroG, KroH.

    M = diag(vec(Kp)) + KroG @ diag(vec(Ke)) @ KroH.T

    where diag(x) constructs a diagnal matrix with vector x on the main diag
    and vec(x) applies COLUMN vectorization to matrix x

    Args:
        Ke (Tensor): (B, P, Q) edge-wise affinity matrix.
            P: number of edges in graph 1, Q: number of edges in graph 2
        Kp (Tensor): (B, M, N) node-wise affinity matrix.
            N: number of nodes in graph 1, M: number of nodes in graph 2
        KroG (List[Tensor]): A list of length B,
            containing (MN, PQ) Kronecker product of G2 and G1
            G1: (B, N, P). G2: (B, M, Q)
        KroH (List[Tensor]): A list of length B,
            containing (MN, PQ) Kronecker product of H2 and H1
            H1: (B, N, P). H2: (B, M, Q)

    Returns:
        Tensor: Affinity Matrix K

    NOTE: This implementation is a workaround for the original
    CSR/CSC sparse implementation with GPU support in the torch version.
    Since Paddle currently does not have adequate support for C++ extensions
    It is dense and devecotrized, and can thus its performance is suboptimal.
    """
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    # NOTE: Every single transpose here matters
    # because we should use a COLUMN VECTORIZATION here.
    res = paddle.zeros((B, M * N, M * N))
    KroG = paddle.stack(KroG)
    KroH = paddle.stack(KroH)
    diag_KroH = paddle.multiply(
        Ke.transpose((0, 2, 1)).reshape([B, -1, 1]),
        KroH.transpose((0, 2, 1)))
    KroG_diag_KroH = paddle.bmm(
        KroG,
        diag_KroH)  # B, MN, MN
    for b in range(B):
        res[b] = paddle.diag(
            Kp[b].transpose((1, 0)).reshape([-1])) + KroG_diag_KroH

    return res
