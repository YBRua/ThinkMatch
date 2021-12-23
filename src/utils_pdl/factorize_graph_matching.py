import paddle
from typing import List
from paddle import Tensor
from paddle.autograd import PyLayer
from src.utils.sparse import bilinear_diag_torch
import numpy as np


def construct_aff_mat_dense(Ke: Tensor, Kp: Tensor, KroG: List[Tensor], KroH: List[Tensor]) -> Tensor:
    """Construct the complete affinity matrix with
    edge-wise affinity matrix Ke, node-wise matrix Kp
    and Kronecker products KroG, KroH.

    M = diag(vec(Kp)) + KroG @ diag(vec(Ke)) @ KroH.T

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
    It is dense and devecotrized, and can thus be SLOW.
    """
    B, P, Q = Ke.shape
    _, N, M = Kp.shape
    res = paddle.zeros((B, M * N, M * N))
    for b in range(B):
        KroG_diag = paddle.matmul(KroG[b], paddle.diag(Ke[b].reshape([-1])))  # MN, PQ
        # print("KroG_diag", KroG_diag.shape)
        KroG_diag_KroH = paddle.matmul(KroG_diag, KroH[b].transpose((1, 0)))  # MN, MN
        # print("KroG-diag-KroH", KroG_diag_KroH.shape)
        res[b] = paddle.diag(Kp[b].reshape([-1])) + KroG_diag_KroH

    return res
