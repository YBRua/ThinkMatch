import paddle
from typing import Sequence, Union


Size = Union[Sequence[int], int]


def swap_axes(
        src: paddle.Tensor,
        axes_1: Size,
        axes_2: Size):
    """Funcitonality the same as `torch.transpose`
    Transposed `axes_1` and `axes_2` of Tensor `src`

    Args:
        src (paddle.Tensor): Input paddle tensor to be transposed
        axes_1 (List[int] or int): An axis
        axes_2 (List[int] or int): Another axis
    """
    if isinstance(axes_1, int):
        axes_1 = [axes_1]
    if isinstance(axes_2, int):
        axes_2 = [axes_2]
    if len(axes_1) != len(axes_2):
        raise ValueError(
            "Axes to be swapped must have the same lengths. "
            + f"Got axes of lengths {len(axes_1)} and {len(axes_2)}")
    dims = [*range(src.dim())]
    for a, b in zip(axes_1, axes_2):
        dims[a], dims[b] = dims[b], dims[a]
    return src.transpose(dims)


def repeat_interleave(
        src: paddle.Tensor,
        repeat: Union[paddle.Tensor, int],
        axis=None):
    """Repeat interleave workaround for paddle,
    since PaddlePaddle currently does not have a `repeat_interleave`

    Functionality the same as `torch.repeat_interleave`

    Args:
        src (paddle.Tensor): Original Tensor
        repeat (Union[paddle.Tensor, int]): number of repeats on one or more axes
        axis ([type], optional): Defaults to None. Axes to perform repeats on.
            If is None, the Tensor `src` will be flattened and repeated

    Example:
        >>> x = paddle.to_tensor([1, 2, 3,])
        >>> repeat_interleave(x, 2)
        >>> [1, 1, 2, 2, 3, 3]
    """
    if axis is None:
        src = src.reshape([-1])
        axis = 0
    dtype = 'int64'
    if isinstance(repeat, paddle.Tensor):
        dtype = repeat.dtype
    repeat = repeat + paddle.zeros([src.shape[axis]], dtype=dtype)
    cumptr = paddle.cumsum(repeat)
    src_sw = swap_axes(src, 0, axis)
    return swap_axes(_cum_repeat_0(src_sw, cumptr), 0, axis)


def _cum_repeat_0(
        src: paddle.Tensor,
        cumptr: paddle.Tensor):
    idx = paddle.scatter(
        paddle.zeros([cumptr[-1]], dtype=cumptr.dtype),
        cumptr,
        paddle.ones_like(cumptr),
        overwrite=False
    )
    visit = paddle.cumsum(idx)
    return src[visit]
