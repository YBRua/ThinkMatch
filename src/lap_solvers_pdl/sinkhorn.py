import paddle
import numpy as np
import paddle.nn as nn
from paddle import Tensor
from src.utils_pdl.pdl_device_trans import place2str


class Sinkhorn(nn.Layer):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4, tau=0.05, log_forward=True):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tau = tau
        self.log_forward = log_forward
        if not log_forward:
            print(
                'Warning: Sinkhorn algorithm not in log scale is deprecated'
                + ' since logrithm is more stable')

    def forward(self, *input, **kwinput):
        if self.log_forward:
            return self.forward_log(*input, **kwinput)
        else:
            return self.forward_ori(*input, **kwinput)  # deprecated

    def forward_log(
            self,
            s,
            nrows=None, ncols=None,
            dummy_row=False,
            dtype=paddle.float32):
        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose((0, 2, 1))
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # operations are performed on log_s
        s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = paddle.concat(
                (s, paddle.full(dummy_shape, -float('inf')).cuda()),
                axis=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        ret_log_s = paddle.full(
            (batch_size, s.shape[1], s.shape[2]),
            -float('inf'), dtype=s.dtype).cuda()
        ret_log_s.stop_gradient = False

        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]))
            col_slice = slice(0, int(ncols[b]))
            log_s = s[b, row_slice, col_slice]

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = paddle.logsumexp(log_s, 0, keepdim=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

        if dummy_row:
            if dummy_shape[1] > 0:
                ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if transposed:
            ret_log_s = ret_log_s.transpose((0, 2, 1))
        if matrix_input:
            ret_log_s.squeeze_(0)

        return paddle.exp(ret_log_s)

    def forward_ori(
            self,
            s,
            nrows=None, ncols=None,
            exp=False, exp_alpha=20,
            dummy_row=False, dtype=paddle.float32):
        batch_size = s.shape[0]

        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = paddle.concat((s, paddle.full(dummy_shape, 0.).cuda()), axis=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = paddle.zeros(
            (batch_size, s.shape[1], s.shape[1]))  # size: row x row
        col_norm_ones = paddle.zeros(
            (batch_size, s.shape[2], s.shape[2]))  # size: col x col
        for b in range(batch_size):
            row_slice = slice(
                0, int(nrows[b]) if nrows is not None else int(s.shape[2]))
            col_slice = slice(
                0, int(ncols[b]) if ncols is not None else int(s.shape[1]))
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = paddle.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                sum = paddle.sum(
                    paddle.multiply(
                        s.unsqueeze(3), col_norm_ones.unsqueeze(1)),
                    axis=2)
            else:
                # row norm
                sum = paddle.sum(
                    paddle.multiply(
                        row_norm_ones.unsqueeze(3), s.unsqueeze(1)),
                    axis=2)

            tmp = paddle.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(
                    0, int(nrows[b]) if nrows is not None else s.shape[2])
                col_slice = slice(
                    0, int(ncols[b]) if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s


class GumbelSinkhorn(nn.Layer):
    """
    Gumbel Sinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    See details in `"Mena et al. Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018"
    <https://arxiv.org/abs/1802.08665>`_

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        This module only supports log-scale Sinkhorn operation.
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, batched_operation=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, batched_operation=batched_operation)

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None,
                sample_num=5, dummy_row=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param sample_num: number of samples
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b m\times n_1 \times n_2)` the computed doubly-stochastic matrix. :math:`m`: number of samples
         (``sample_num``)

        The samples are stacked at the fist dimension of the output tensor. You may reshape the output tensor ``s`` as:

        ::

            s = torch.reshape(s, (-1, sample_num, s.shape[1], s.shape[2]))

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = paddle.empty_like(t_like).uniform_()
            return -paddle.log(-paddle.log(u + eps) + eps)

        s_rep = paddle.to_tensor(np.repeat(s, sample_num, axis=0))
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep = paddle.to_tensor(np.repeat(nrows, sample_num, axis=0))
        ncols_rep = paddle.to_tensor(np.repeat(ncols, sample_num, axis=0))
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row)
        return s_rep
