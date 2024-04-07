import torch.nn.functional as F
import torch.nn as nn
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance, SNRDistance


def get_loss(params):
    margin = params['margin']
    neg_margin = 1-margin
    if params['dist'] == 'L1':
        distance = LpDistance(p=1)
    elif params['dist'] == 'L2':
        distance = LpDistance(normalize_embeddings=True, p=2, power=1)
    elif params['dist'] == 'SNR':
        distance = SNRDistance()

    if params['loss'] == 'CF':
        loss = losses.CosFaceLoss(8, params['output'], margin=0.35, scale=64)

    elif params['loss'] == 'NCA':
        loss = losses.NCALoss(softmax_scale=1)

    elif params['loss'] == 'CL':
        # loss = ContrastiveLoss(margin=margin)
        # loss = losses.ContrastiveLoss(pos_margin=margin, neg_margin= neg_margin, distance = distance)
        loss = losses.ContrastiveLoss()

    elif params['loss'] == 'OCL':
        pass

    elif params['loss'] == 'TR':
        loss = TripletLoss(margin=margin)
    
    elif params['loss'] == 'TML':
        loss = losses.TripletMarginLoss(margin=margin, distance = distance)

    elif params['loss'] == 'MSL':
        loss = losses.MultiSimilarityLoss()

    else:
        raise ValueError(f"{params['loss']} is not a viable loss option")

    return loss


class ContrastiveLoss(nn.Module):
    """
    Takes embeddings of two samples and a target label == 1 if
    samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1.0 - target).float() * F.relu(self.margin
                                                        - (distances + self.eps).sqrt()).pow(2))
        # sqrt() of a tiny number may be negative!
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
