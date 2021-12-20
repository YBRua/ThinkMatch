import paddle
import paddle.nn as nn


class Voting(nn.Layer):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(axis=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
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
