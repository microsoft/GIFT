"""
Negative-pair policy shared by every model (GNN phases 1-3 and RF) and the evaluator.

A fixed random 20% of all ChEMBL35 compound x protein pairs is reserved for evaluation:
validation/test negatives are drawn only from it, and no training step may use it as a negative.
Training negatives exclude training positives only, so held-out positives are never singled out.
"""

import numpy as np

RESERVE_FRACTION = 0.2
RESERVE_SEED = 20251


def reserved_eval_mask(num_compounds, num_proteins):
    """Bool [num_compounds, num_proteins]; True = pair reserved for evaluation negatives."""
    return np.random.default_rng(RESERVE_SEED).random((num_compounds, num_proteins)) < RESERVE_FRACTION


class _Exclusion:
    """Set-like `pair in excl` test used by the negative samplers."""

    def __init__(self, pairs, reserved, block_reserved):
        self.pairs = set(pairs)
        self.reserved = reserved
        self.block_reserved = block_reserved

    def __contains__(self, pair):
        c, p = pair
        return pair in self.pairs or bool(self.reserved[c, p]) == self.block_reserved


def training_exclusion(train_pairs, reserved):
    """Training negatives: anything except training positives and reserved pairs."""
    return _Exclusion(train_pairs, reserved, block_reserved=True)


def evaluation_exclusion(known_positives, reserved):
    """Evaluation negatives: reserved pairs that are not known ChEMBL35 positives."""
    return _Exclusion(known_positives, reserved, block_reserved=False)


def sample_negatives(rng, compound_pool, num_proteins, n, exclude):
    out, seen = [], set()
    while len(out) < n:
        c = rng.choice(compound_pool, size=2 * (n - len(out)))
        p = rng.randint(0, num_proteins, size=c.size)
        for pair in zip(c.tolist(), p.tolist()):
            if pair not in exclude and pair not in seen:
                out.append(pair)
                seen.add(pair)
                if len(out) == n:
                    break
    return out


def dti_random_negatives(seed, potent_pairs, num_proteins, n, exclude):
    """Random negatives over the dataset's drug set, i.e. compounds with >= 1 known potent interaction (as in standard DTI benchmarks)."""
    pool = np.asarray(sorted({int(c) for c, _ in potent_pairs}))
    return sample_negatives(np.random.RandomState(seed), pool, num_proteins, n, exclude)


def sample_compound_matched_negatives(rng, positives, num_proteins, ratio, exclude):
    """Per compound c: `ratio` x (#positives of c) negatives (c, p'), drawn without replacement and
    capped at the number of eligible proteins (a few heavily tested compounds hit the cap)."""
    counts = {}
    for c, _ in positives:
        counts[int(c)] = counts.get(int(c), 0) + 1
    out = []
    for c in sorted(counts):
        eligible = [p for p in range(num_proteins) if (c, p) not in exclude]
        k = min(ratio * counts[c], len(eligible))
        out.extend((c, int(p)) for p in rng.choice(eligible, size=k, replace=False))
    return out


def sample_protein_matched_negatives(rng, positives, compound_pool, ratio, exclude):
    """Per protein p: `ratio` x (#positives of p) negatives (c', p), c' drawn without replacement from
    `compound_pool` (the compounds with a known potent interaction) and capped at the eligible ones."""
    counts = {}
    for _, p in positives:
        counts[int(p)] = counts.get(int(p), 0) + 1
    out = []
    for p in sorted(counts):
        eligible = [int(c) for c in compound_pool if (int(c), p) not in exclude]
        k = min(ratio * counts[p], len(eligible))
        out.extend((int(c), p) for c in rng.choice(eligible, size=k, replace=False))
    return out
