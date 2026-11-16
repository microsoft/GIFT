"""Phases 1-2 of GIFT (contrastive pretraining with hard negatives, then soft-edge learning) for one run."""

import train_backbone_common as base
from contrastive_loss import TripletLossRandomCandidates

base.TripletContrastiveLoss = TripletLossRandomCandidates

if __name__ == "__main__":
    base.main()
