import paddle
import paddle.nn as nn


class Voting(nn.Layer):
    def __init__(self, alpha=200, pixel_thresh=None):
        """Voting Layer
        Computes a new row-stotatic matrix with softmax.
        A large number (`alpha`) is multiplied to the input stochastic matrix to scale up the difference.

        Args:
            `alpha` (int, optional): Value multiplied before softmax.
                Used for scaling up the variance
                Default: 200.
            `pixel_thresh` (optional): Threshold for computing displacement.
                Will ignore points within this threshold.
                But CURRENTLY IT IS NOT IMPLEMENTED.
                Default: None.
        """
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(axis=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        """Forward pass of the voting layer.

        Args:
            `s` (Tensor): permutation matrix or doubly stochastic matrix
            `nrow_gt`: [description]
            `ncol_gt` (optional): [description]. Defaults to None.

        Returns:
            `ret_s`: Softmax matrix
        """
        ret_s = paddle.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(self.alpha * s[b, 0:n, :])
            else:
                tmp = int(ncol_gt[b].numpy())
                ret_s[b, 0:int(n), 0:tmp] =\
                    self.softmax(self.alpha * s[b, 0:int(n), 0:tmp])

        return ret_s
