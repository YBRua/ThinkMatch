import paddle
from paddle import Tensor


def geo_edge_feature(P: Tensor, G: Tensor, H: Tensor, norm_d=256):
    """Compute geometric edge features [d, cos(theta), sin(theta)]
    Adjacency matrix is formed by A = G * H^T

    Args:
        P (Tensor): Set of keypoints (B, N, 2)
        G (Tensor): factorized graph partition G (B, N, E)
        H (Tensor): factorized graph partition H (B, N, E)
        norm_d (int, optional): normalize Euclidean distance by norm_d.
            Defaults to 256.

    Returns:
        [type]: [description]
    """
    # b, num_edges, dim
    p1 = paddle.sum(
        paddle.multiply(P.unsqueeze(-2), G.unsqueeze(-1)), axis=1)
    p2 = paddle.sum(
        paddle.multiply(P.unsqueeze(-2), H.unsqueeze(-1)), axis=1)

    # b, num_edges
    d = paddle.norm(
        (p1 - p2) / (norm_d * paddle.sum(G, dim=1)).unsqueeze(-1),
        axis=-1)  # non-existant elements are nan

    # non-existant elements are nan
    cos_theta = (p1[:, :, 0] - p2[:, :, 0]) / (d * norm_d)
    sin_theta = (p1[:, :, 1] - p2[:, :, 1]) / (d * norm_d)

    return paddle.stack((d, cos_theta, sin_theta), axis=1)
