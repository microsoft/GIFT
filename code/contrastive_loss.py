"""Triplet contrastive loss of the GIFT encoder: TripletContrastiveLoss with hard negatives that share the
compound or the protein of a training interaction, drawn as a random sample of the eligible pairs.
"""

import random

import torch
import torch.nn.functional as F

from model_encoder import TripletContrastiveLoss

CANDIDATES_PER_SOURCE = 50
RANDOM_CANDIDATES = 30
POSITIVES_PER_ANCHOR = 20


class TripletLossRandomCandidates(TripletContrastiveLoss):

    def _pair_embedding_loss_advanced(self, z_compounds, z_proteins, positive_pairs, device):
        positive_set = set(positive_pairs)
        reserved = getattr(self, "reserved_mask", None)

        def blocked(c, p):
            return (c, p) in positive_set or (reserved is not None and bool(reserved[c, p]))

        batch_pairs = random.sample(positive_pairs, min(self.batch_size, len(positive_pairs)))
        anchor_c = torch.tensor([c for c, _ in batch_pairs], device=device)
        anchor_p = torch.tensor([p for _, p in batch_pairs], device=device)
        anchor_embs = F.normalize(
            self.create_pair_embedding_batch(z_compounds[anchor_c], z_proteins[anchor_p]), p=2, dim=1
        )

        num_compounds, num_proteins = z_compounds.size(0), z_proteins.size(0)
        neg_embs_list = []
        for i, (c, p) in enumerate(batch_pairs):
            same_compound = [k for k in range(num_proteins) if k != p and not blocked(c, k)]
            same_protein = [l for l in range(num_compounds) if l != c and not blocked(l, p)]
            cand = [(c, k) for k in random.sample(same_compound, min(CANDIDATES_PER_SOURCE, len(same_compound)))]
            cand += [(l, p) for l in random.sample(same_protein, min(CANDIDATES_PER_SOURCE, len(same_protein)))]
            for _ in range(RANDOM_CANDIDATES):
                nc, np_ = random.randint(0, num_compounds - 1), random.randint(0, num_proteins - 1)
                if not blocked(nc, np_):
                    cand.append((nc, np_))

            if not cand:
                neg_embs_list.append(torch.zeros(1, anchor_embs.size(1), device=device))
                continue
            cc = torch.tensor([x for x, _ in cand], device=device)
            cp = torch.tensor([y for _, y in cand], device=device)
            cand_embs = F.normalize(self.create_pair_embedding_batch(z_compounds[cc], z_proteins[cp]), p=2, dim=1)
            sims = (cand_embs @ anchor_embs[i]).detach()
            order = torch.argsort(sims, descending=True).tolist()

            if self.hard_negative_mining:
                num_hard = min(int(self.max_negatives_per_positive * 0.7), len(order))
                chosen = order[:num_hard]
                rest = order[num_hard:]
                chosen += random.sample(rest, min(self.max_negatives_per_positive - num_hard, len(rest)))
            else:
                chosen = random.sample(order, min(self.max_negatives_per_positive, len(order)))
            neg_embs_list.append(cand_embs[chosen[: self.max_negatives_per_positive]])

        max_negs = max(e.size(0) for e in neg_embs_list)
        neg_embs = torch.zeros(len(batch_pairs), max_negs, anchor_embs.size(1), device=device)
        for i, e in enumerate(neg_embs_list):
            neg_embs[i, : e.size(0)] = e

        pos_pairs = []
        for c, p in batch_pairs:
            others = [q for q in positive_pairs if q != (c, p)]
            if others:
                pos_pairs.extend(random.sample(others, min(POSITIVES_PER_ANCHOR, len(others))))
        if pos_pairs:
            pc = torch.tensor([c for c, _ in pos_pairs], device=device)
            pp = torch.tensor([p for _, p in pos_pairs], device=device)
            pos_embs = F.normalize(self.create_pair_embedding_batch(z_compounds[pc], z_proteins[pp]), p=2, dim=1)
        else:
            pos_embs = anchor_embs

        pos_sims = (anchor_embs @ pos_embs.t()) / self.temperature
        neg_sims = torch.bmm(anchor_embs.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1) / self.temperature
        loss = torch.clamp(
            self.margin + neg_sims.mean(dim=1) * self.neg_weight - pos_sims.mean(dim=1) * self.pos_weight,
            min=1e-4,
        )
        return loss.mean()
