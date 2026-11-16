"""Point 6: deterministic, label-free domain-preserving crops of protein sequences for the DTA baselines.

For every protein of data/protein_sequences.json longer than the model's sequence limit L
(DeepDTA/GraphDTA L=1000, DrugBAN L=1200) a contiguous window of L residues is kept:
  * domains = all Pfam-A domains of the target's matching ChEMBL 36 component (component_domains, 1-based inclusive);
  * if the union of domains (first start .. last end) fits in L, the window is centred on that union
    (clipped to the sequence ends);
  * otherwise the window maximising domain coverage is kept, lexicographically: first the residues of protein-kinase
    domains (Pfam PF00069 Pkinase / PF07714 Pkinase_Tyr; the ligand-binding domain of the driver kinases), then the
    residues of any Pfam-A domain (ties: the first contiguous block of maximal windows, middle start);
  * proteins without a domain annotation (or without a ChEMBL component whose sequence matches) keep the
    N-terminal prefix, as in the original scripts.
Component matching: identical sequence first, else equal length + single component, else none.
Writes domain_crops.json ({"1000": {target: {start, end, seq, rule}}, "1200": ...}) and crop_report.json.
"""
import os as _os, sys as _sys  # release bootstrap: run from any directory as python code/robustness/<script>.py
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_sys.path[:0] = [_os.path.join(_REPO, "code"), _os.path.join(_REPO, "code", "robustness")]
_os.chdir(_REPO)
import json
import os
import sqlite3

ROOT = _REPO
OUT = os.path.join(ROOT, "results/robustness/domain_crop")
DB = os.path.join(ROOT, "data/chembl_db/chembl_36/chembl_36_sqlite/chembl_36.db")
LIMITS = (1000, 1200)
PK = ("PF00069", "PF07714")
DRIVERS = {"CHEMBL203": "EGFR", "CHEMBL1824": "ERBB2", "CHEMBL4247": "ALK", "CHEMBL5568": "ROS1",
           "CHEMBL2041": "RET", "CHEMBL3717": "MET", "CHEMBL5145": "BRAF", "CHEMBL2815": "NTRK1",
           "CHEMBL4898": "NTRK2", "CHEMBL5608": "NTRK3"}


def covered(doms, s, e):
    """Residues of [s, e] (1-based inclusive) covered by the union of the domain intervals."""
    cov = set()
    for a, b, _ in doms:
        cov.update(range(max(a, s), min(b, e) + 1))
    return len(cov)


def choose_window(n, L, doms):
    if not doms:
        return 1, "prefix_no_domain"
    a, b = min(d[0] for d in doms), max(d[1] for d in doms)
    if b - a + 1 <= L:
        s = int((a + b) // 2 - (L - 1) // 2)
        return max(1, min(s, n - L + 1)), "centered_union"
    mask = [0] * (n + 2)
    for x, y, src in doms:
        for i in range(x, min(y, n) + 1):
            mask[i] = max(mask[i], (n + 1) if src.split(":")[0] in PK else 1)
    pref = [0] * (n + 1)
    for i in range(1, n + 1):
        pref[i] = pref[i - 1] + mask[i]
    cov = [pref[s + L - 1] - pref[s - 1] for s in range(1, n - L + 2)]
    best = max(cov)
    first = cov.index(best)
    last = first
    while last + 1 < len(cov) and cov[last + 1] == best:
        last += 1
    return (first + last) // 2 + 1, "max_coverage"


def main():
    seqs = json.load(open(os.path.join(ROOT, "data/protein_sequences.json")))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    comps = {}
    for t, cid, seq in con.execute("""SELECT td.chembl_id, cs.component_id, cs.sequence FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid JOIN component_sequences cs ON tc.component_id = cs.component_id"""):
        comps.setdefault(t, []).append((cid, seq or ""))
    doms = {}
    for cid, s, e, src, name in con.execute("""SELECT cd.component_id, cd.start_position, cd.end_position,
            d.source_domain_id, d.domain_name FROM component_domains cd JOIN domains d ON cd.domain_id = d.domain_id"""):
        doms.setdefault(cid, []).append((int(s), int(e), f"{src}:{name}"))
    con.close()

    crops, report = {str(L): {} for L in LIMITS}, {}
    match_kind = {}
    for t, seq in seqs.items():
        cs = comps.get(t, [])
        exact = [c for c in cs if c[1] == seq]
        same_len = [c for c in cs if len(c[1]) == len(seq)]
        if exact:
            c, kind = exact[0], "exact_sequence"
        elif len(cs) == 1 and same_len:
            c, kind = same_len[0], "same_length"
        else:
            c, kind = None, "no_matching_component"
        d = sorted(x for x in doms.get(c[0], []) if x[1] <= len(seq)) if c else []
        match_kind[t] = kind
        for L in LIMITS:
            if len(seq) <= L:
                continue
            s, rule = choose_window(len(seq), L, d)
            e = s + L - 1
            pk = [x for x in d if x[2].split(":")[0] in PK]
            crops[str(L)][t] = {"start": s, "end": e, "length": len(seq), "rule": rule, "seq": seq[s - 1:e],
                                "domains": [f"{x[2]}:{x[0]}-{x[1]}" for x in d],
                                "domain_residues_total": covered(d, 1, len(seq)),
                                "domain_residues_kept": covered(d, s, e),
                                "domain_residues_kept_prefix": covered(d, 1, L),
                                "pk_residues_total": covered(pk, 1, len(seq)),
                                "pk_residues_kept": covered(pk, s, e),
                                "pk_residues_kept_prefix": covered(pk, 1, L)}
    for L in LIMITS:
        c = crops[str(L)]
        rules = {}
        for v in c.values():
            rules[v["rule"]] = rules.get(v["rule"], 0) + 1
        pk_prot = [v for v in c.values() if v["pk_residues_total"] > 0]
        report[str(L)] = {
            "n_cropped": len(c), "rules": rules,
            "n_no_domain_prefix_kept": rules.get("prefix_no_domain", 0),
            "domain_residue_coverage_prefix": sum(v["domain_residues_kept_prefix"] for v in c.values())
            / max(1, sum(v["domain_residues_total"] for v in c.values())),
            "domain_residue_coverage_crop": sum(v["domain_residues_kept"] for v in c.values())
            / max(1, sum(v["domain_residues_total"] for v in c.values())),
            "n_with_kinase_domain": len(pk_prot),
            "n_kinase_domain_complete_prefix": sum(v["pk_residues_kept_prefix"] == v["pk_residues_total"] for v in pk_prot),
            "n_kinase_domain_complete_crop": sum(v["pk_residues_kept"] == v["pk_residues_total"] for v in pk_prot),
            "drivers": {g: ({k: v for k, v in c[t].items() if k != "seq"} if t in c else
                            {"length": len(seqs[t]), "rule": "not_cropped (length <= limit)"})
                        for t, g in DRIVERS.items()},
        }
    report["component_match"] = {k: sum(v == k for v in match_kind.values())
                                 for k in ("exact_sequence", "same_length", "no_matching_component")}
    report["component_match_among_gt1000"] = {k: sum(match_kind[t] == k for t in crops["1000"])
                                              for k in ("exact_sequence", "same_length", "no_matching_component")}
    report["n_proteins"] = len(seqs)
    json.dump(crops, open(os.path.join(OUT, "domain_crops.json"), "w"))
    json.dump(report, open(os.path.join(OUT, "crop_report.json"), "w"), indent=1)
    for L in LIMITS:
        r = report[str(L)]
        print(L, {k: v for k, v in r.items() if k != "drivers"})
        for g, v in r["drivers"].items():
            print("  ", g, v.get("length"), v.get("rule"), v.get("start"), v.get("end"), v.get("domains", ""),
                  "pk kept", v.get("pk_residues_kept"), "/", v.get("pk_residues_total"), "prefix", v.get("pk_residues_kept_prefix"))
    print(report["component_match"], report["component_match_among_gt1000"])


if __name__ == "__main__":
    main()
