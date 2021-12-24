import paddle
import paddle.nn as nn
# TODO: We use dense implementation now
# from utils.sparse import sbmm


class PowerIteration(nn.Layer):
    def __init__(self, max_iter=50, stop_thresh=2e-7):
        super(PowerIteration, self).__init__()
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh

    def forward(self, M, v0=None):
        """Power iteration layer.
        Computes the leading eigenvector of input matrix by Power Iteration.

        `v_{k+1} = M * v_{k} / ||M * v_{k}||_2`

        Args:
            `M` (Tensor): Input tensor
            `v0` (Tensor, optional): Initial eigenvector.
                If is None, it will be initialized with `ones`.
                Default: None.

        Returns:
            `v`: Leading eigenvector
        """
        batch_num = M.shape[0]
        mn = M.shape[1]
        if v0 is None:
            v0 = paddle.ones((batch_num, mn, 1), dtype=M.dtype)

        v = vlast = v0
        for i in range(self.max_iter):
            # No more sparse currently
            # if M.is_sparse:
            #     v = sbmm(M, v)
            # else:
            v = paddle.bmm(M, v)
            n = paddle.norm(v, p=2, axis=1)
            v = paddle.matmul(v, (1 / n).reshape([batch_num, 1, 1]))
            if paddle.norm(v - vlast) < self.stop_thresh:
                return v.reshape([batch_num, -1])
            vlast = v

        return v.reshape([batch_num, -1])
