"""Random-augmentation control: the soft-edge Phase 2, but edges are chosen UNIFORMLY AT RANDOM.

This isolates "learning WHICH edges to add" from "adding edges at all". Phase 2 is identical to the full
model (same loss, epochs, checkpoint selection, and the same NUMBER and TYPE of added edges per epoch --
1000*(epoch+1) compound-protein, 100*(epoch+1) protein-protein, and 100*(epoch+1) compound-compound), but
the edges are selected at random instead of by the learned attention. Each random edge reuses the weight
magnitude the learned top-k would have had, so the augmentation strength is matched and only the SELECTION
differs. Use with --resume_phase2 and the Phase 1 checkpoint of the full run (shared Phase 1 encoder),
exactly as the no-soft-edge ablation.
"""

import torch

import model_encoder as gm
import train_backbone_common as base
from contrastive_loss import TripletLossRandomCandidates

base.TripletContrastiveLoss = TripletLossRandomCandidates


def _random_cp(self, attention_scores, epoch, num_compounds, num_proteins):
    device = attention_scores.device
    flat = attention_scores.flatten()
    k = min(1000 * (epoch + 1), flat.size(0))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
    top_scores, _ = torch.topk(flat, k, sorted=False)              # matched weight magnitudes
    idx = torch.randperm(flat.size(0), device=device)[:k]          # random selection
    c = idx // num_proteins
    p = idx % num_proteins + num_compounds
    ei = torch.stack([torch.cat([c, p]), torch.cat([p, c])], dim=0)
    ew = torch.cat([top_scores, top_scores])
    return ei, ew


def _random_homogeneous(self, attention_scores, epoch, num_nodes, node_offset=0):
    device = attention_scores.device
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
    triu = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    triu_scores = attention_scores[triu[0], triu[1]]
    k = min(100 * (epoch + 1), triu_scores.size(0))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
    top_scores, _ = torch.topk(triu_scores, k, sorted=False)       # matched weight magnitudes
    sel = torch.randperm(triu_scores.size(0), device=device)[:k]   # random selection
    i = triu[0][sel] + node_offset
    j = triu[1][sel] + node_offset
    ei = torch.stack([torch.cat([i, j]), torch.cat([j, i])], dim=0)
    ew = torch.cat([top_scores, top_scores])
    return ei, ew


gm.SoftEdgeMask.select_edges_progressive = _random_cp
gm.HomogeneousSoftEdgeMask.select_edges_progressive = _random_homogeneous

if __name__ == "__main__":
    base.main()
