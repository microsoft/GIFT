"""Soft-edge ablation of the GIFT backbone: Phase 2 runs with the same loss, epochs, and checkpoint selection,
but no proposed edges are added (apply_edge_mask forced to False). Use with --resume_phase2 and the Phase 1
checkpoint of the full run, so both arms share the same Phase 1 encoder."""

import model_encoder as gm
import train_backbone_common as base
from contrastive_loss import TripletLossRandomCandidates

base.TripletContrastiveLoss = TripletLossRandomCandidates
_base_forward = gm.DrugTargetGNN.forward


def _forward_without_soft_edges(self, *args, **kwargs):
    kwargs["apply_edge_mask"] = False
    return _base_forward(self, *args, **kwargs)


gm.DrugTargetGNN.forward = _forward_without_soft_edges

if __name__ == "__main__":
    base.main()
