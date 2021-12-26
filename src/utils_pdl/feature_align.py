import paddle
import paddle.nn.functional as F
from paddle import Tensor
from .pdl_device_trans import place2int


def feature_align_fast(raw_feature, P, ns_t, ori_size):
    """Faster feature align with bi-linear interpolation.
    Speeds up operation by vectorization

    Implementation based on @eliphatfs.

    Args:
        raw_feature: Raw input features for keypoints
        P: Set of keypoints
        ns_t: Number of nodes. Features beyond this range will be set to 0
        ori_size: Rescaling factor

    Returns:
        F_: Aligned features
    """
    F_ = F.grid_sample(
        raw_feature,
        2 * P.unsqueeze(-2) / ori_size[0] - 1,
        'bilinear',
        'border',
        align_corners=False
    ).squeeze(-1)
    for b, ns in enumerate(ns_t):
        F_[b, :, ns:] = 0
    return F_


def feature_align(
        raw_feature: Tensor,
        P: Tensor,
        ns_t: Tensor,
        ori_size: tuple,
        device=None):
    """Perform feature align from the raw feature map.

    Args:
        `raw_feature` (Tensor): raw feature map
        `P` (Tensor): point set containing point coordinates
        `ns_t` (Tensor): number of exact points in the point set
        `ori_size` (tuple): size of the original image
        `device` (optional): device. If not specified, it will be the same as the input.
            Default: None.

    Returns:
        `F`
    """
    if device is None:
        device = raw_feature.place

    batch_num = raw_feature.shape[0]
    channel_num = raw_feature.shape[1]
    n_max = P.shape[1]
    # n_max = 0
    # for idx in range(batch_num):
    #     n_max = max(ns_t[idx], n_max)

    ori_size = paddle.to_tensor(ori_size, dtype='float32', place=device)
    F_ = paddle.zeros(
        [batch_num, channel_num, n_max],
        dtype='float32').cuda(place2int(device))
    F_.stop_gradient = False
    for idx, feature in enumerate(raw_feature):
        n = int(ns_t[idx].numpy())
        feat_size = paddle.to_tensor(
            feature.shape[1:3],
            dtype='float32',
            place=device)
        _P = P[idx, 0:n]
        F_[idx, :, 0:n] = interp_2d(
            feature,
            _P,
            ori_size,
            feat_size,
            out=F_[idx, :, 0:n])
        # interp_2d(feature, _P, ori_size, feat_size, out=F[idx, :, 0:n])
        # F[idx, :, 0:n] += interp_2d(feature, _P, ori_size, feat_size)
    return F_


def interp_2d(
        z: Tensor,
        P: Tensor,
        ori_size: Tensor,
        feat_size: Tensor,
        out=None,
        device=None):
    """
    Interpolate in 2d grid space. z can be 3-dimensional where the 3rd dimension is feature vector.
    :param z: 2d/3d feature map
    :param P: input point set
    :param ori_size: size of the original image
    :param feat_size: size of the feature map
    :param out: optional output tensor
    :param device: device. If not specified, it will be the same as the input
    :return: F
    """
    if device is None:
        device = z.place

    step = ori_size / feat_size
    if out is None:
        out = paddle.zeros(
            [z.shape[0], P.shape[0]],
            dtype='float32').cuda(place2int(device))
    for i, p in enumerate(P):
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate_paddle(z, p[0], p[1])

    return out


def bilinear_interpolate_paddle(
        im: Tensor,
        x: Tensor,
        y: Tensor,
        out=None,
        device=None):
    """
    Bi-linear interpolate 3d feature map im to 2d plane (x, y)
    :param im: 3d feature map
    :param x: x coordinate
    :param y: y coordinate
    :param out: optional output tensor
    :param device: device. If not specified, it will be the same as the input
    :return: interpolated feature vector
    """
    if device is None:
        device = im.place
    x = paddle.to_tensor(x, dtype='float32', place=device)
    y = paddle.to_tensor(y, dtype='float32', place=device)

    x0 = paddle.floor(x)
    x1 = x0 + 1
    y0 = paddle.floor(y)
    y1 = y0 + 1

    x0 = paddle.clip(x0, 0, im.shape[2] - 1)
    x1 = paddle.clip(x1, 0, im.shape[2] - 1)
    y0 = paddle.clip(y0, 0, im.shape[1] - 1)
    y1 = paddle.clip(y1, 0, im.shape[1] - 1)

    x0 = paddle.to_tensor(x0, dtype='int32', place=device)
    x1 = paddle.to_tensor(x1, dtype='int32', place=device)
    y0 = paddle.to_tensor(y0, dtype='int32', place=device)
    y1 = paddle.to_tensor(y1, dtype='int32', place=device)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    # to perform nearest neighbor interpolation if out of bounds
    if x0 == x1:
        if x0 == 0:
            x0 -= 1
        else:
            x1 += 1
    if y0 == y1:
        if y0 == 0:
            y0 -= 1
        else:
            y1 += 1

    x0 = paddle.to_tensor(x0, dtype='float32', place=device)
    x1 = paddle.to_tensor(x1, dtype='float32', place=device)
    y0 = paddle.to_tensor(y0, dtype='float32', place=device)
    y1 = paddle.to_tensor(y1, dtype='float32', place=device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out
