import paddle
import paddle.nn as nn
# TODO: We use dense implementation now
# from utils.sparse import sbmm


class PowerIteration(nn.Layer):
    """
    Power Iteration layer to compute the leading eigenvector of input matrix. The idea is from Spectral Graph Matching.
    For every iteration,
        v_k+1 = M * v_k / ||M * v_k||_2
    Parameter: maximum iteration max_iter
    Input: input matrix M
           (optional) initialization vector v0. If not specified, v0 will be initialized with all 1.
    Output: computed eigenvector v
    """
    def __init__(self, max_iter=50, stop_thresh=2e-7):
        super(PowerIteration, self).__init__()
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh

    def forward(self, M, v0=None):
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
