import torch
import torch.nn as nn


class HeatmapWeightedLoss(nn.Module):
    def __init__(self, class_weights=(1.0, 1.0, 1.0), normalize_weights=False):
        super(HeatmapWeightedLoss, self).__init__()
        weights_torched = torch.tensor(class_weights, requires_grad=False).unsqueeze(0)
        if normalize_weights:
            weights_torched /= weights_torched.sum()
        weights_torched = weights_torched.contiguous()

        self.weights = nn.Parameter(weights_torched, requires_grad=False)

    def forward(self, heatmaps_gt, heatmaps_pred):
        raise NotImplementedError()


class HeatmapMSE(HeatmapWeightedLoss):
    def forward(self, heatmaps_gt, heatmaps_pred):
        diffs_powered = torch.pow((heatmaps_gt - heatmaps_pred), 2)
        means = diffs_powered.mean(axis=(2, 3)) * self.weights
        return means.mean()


class HeatmapMAE(HeatmapWeightedLoss):
    def forward(self, heatmaps_gt, heatmaps_pred):
        diffs = torch.abs(heatmaps_gt - heatmaps_pred)
        means = diffs.mean(axis=(2, 3)) * self.weights
        return means.mean()


class HeatmapHuber(HeatmapWeightedLoss):
    def __init__(self, class_weights, normalize_weights=False, delta=1.0):
        super(HeatmapHuber, self).__init__(class_weights, normalize_weights)
        self.delta = delta
        self.c0 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.c1 = nn.Parameter(torch.tensor(0.5 * delta), requires_grad=False)
        self.c2 = nn.Parameter(torch.tensor(0.5 * delta * delta), requires_grad=False)

    def forward(self, heatmaps_gt, heatmaps_pred):
        diffs = torch.abs(heatmaps_gt - heatmaps_pred)
        squares = torch.mul(self.c0, torch.pow(heatmaps_gt - heatmaps_pred, 2))
        linears = torch.add(torch.mul(self.c1, diffs), self.c2)
        hubers = torch.where(diffs < self.delta, squares, linears)
        means = hubers.mean(axis=(2, 3)) * self.weights
        return means.mean()


class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmaps.

    Parameters
    ----------
    alpha: float
        A balanced form for Focal loss.
    gamma: float
        The gamma for calculating the modulating factor.
    loss_weight:
        The weight for the Focal loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0, loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.gamma)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return self.loss_weight * loss
