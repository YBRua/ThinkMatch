import paddle
import paddle.nn as nn


class Displacement(nn.Layer):
    """Displacement Layer."""
    def __init__(self):
        super(Displacement, self).__init__()

    def forward(self, s, P_src, P_tgt, ns_gt=None):
        """Computes the displacement vector for each point in the source image,
        with its corresponding point (or points) in target image.

        `d = s * P_tgt - P_src`

        The output is a displacement matrix `d` constructed from displacement vectors.
        This metric measures the shift from source point to predicted target point,
        and can be applied for matching accuracy.

        In addition to the displacement matrix `d`,
        this function will also return a gradient mask `grad_mask`,
        which helps filter out dummy nodes in practice.

        Args:
            `s` (Tensor): [description]
            `P_src` (Tensor): Keypoint Tensro of source graph
            `P_tgt` (Tensor): Keypoint Tensor of target graph
            `ns_gt` (int, optional): Ground truth number of nodes. Default: None.

        Returns:
            `d`: Displacement matrix
            `grad_mask`: Gradient mask for filtering out dummy nodes
        """
        if ns_gt is None:
            max_n = s.shape[1]
            P_src = P_src[:, 0:max_n, :]
            grad_mask = None
        else:
            grad_mask = paddle.zeros_like(P_src)
            for b, n in enumerate(ns_gt):
                grad_mask[b, 0:n] = 1

        d = paddle.matmul(s, P_tgt) - P_src
        return d, grad_mask
