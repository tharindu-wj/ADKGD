"""Frozen link-predictor scorer loaded from published LibKGE checkpoints.

Option B (see experiments/docs/OPTION_B_PLAN.md): the plausibility signal for
close-but-false negative sampling is a *pretrained, citable* ComplEx -- the
ICLR-2020 "You CAN Teach an Old Dog New Tricks!" best-config checkpoints
(Ruffinelli, Broscheit & Gemulla). We load raw tensors only: no libkge
install (env is Python 3.13), no training, no fine-tuning. This module is
kgsage-internal and ADKGD-agnostic.

Verified checkpoint facts (probed 2026-07-03):
  fb15k-237-complex.pt : reciprocal-relations model. entity emb (14541, 256),
                         relation emb (474, 256) = 237 base + 237 inverse
                         (inverse index = base + 237). Entity/relation ids are
                         FIRST-APPEARANCE order over the LibKGE archive's
                         train.txt only.
  wnrr-complex.pt      : plain ComplEx. entity emb (40943, 128), relation emb
                         (11, 128). Ids are first-appearance over
                         train+valid+test (40943 > train-only 40559).
The loader auto-detects both properties by matching map sizes against the
checkpoint tensors, and `filtered_mrr()` is the correctness gate: it must
reproduce the published test MRR (0.348 FB15K-237 / 0.475 WN18RR) on the
archive splits before any negative sampling trusts these scores.

LibKGE ComplEx score (kge/model/complex.py, halves layout re|im):
  score(s,p,o) = Re(<s, p, conj(o)>)
which factorises to one matmul per direction:
  tails:  u = s (x) p            ; score(., o) = u . [o_re | o_im]
  heads:  v = conj(p) (x) o *    ; score(s, .) = v . [s_re | s_im]
  (reciprocal models instead score heads as tails of (o, p + n_rel)).
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path

import torch


# --------------------------------------------------------------------------
# checkpoint loading without libkge installed
# --------------------------------------------------------------------------

class _Stub:
    """Placeholder for unpicklable libkge objects (Config etc.)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        object.__setattr__(self, "_stub_state", state)


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (_Stub,), {"__module__": module})


def _load_raw_checkpoint(ckpt_path: str | Path) -> dict:
    shim = types.ModuleType("kge_stub_pickle")
    shim.Unpickler = _StubUnpickler
    shim.load = lambda f, **kw: _StubUnpickler(f, **kw).load()
    return torch.load(str(ckpt_path), pickle_module=shim,
                      weights_only=False, map_location="cpu")


def _first_appearance_maps(split_paths: list[Path]) -> tuple[dict, dict]:
    """LibKGE preprocess_default id assignment: scan lines in order, assign
    len(map) on first appearance -- subject, then predicate, then object."""
    ent, rel = {}, {}
    for path in split_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                s, p, o = parts
                if s not in ent:
                    ent[s] = len(ent)
                if p not in rel:
                    rel[p] = len(rel)
                if o not in ent:
                    ent[o] = len(ent)
    return ent, rel


class ComplExScorer:
    """Frozen ComplEx over LibKGE tensors, string-keyed.

    All score functions are raw (uncalibrated across relations); callers who
    need cross-relation comparability should rank within a candidate pool
    (Option B band rule) rather than compare absolute scores.
    """

    def __init__(self, ent_emb: torch.Tensor, rel_emb: torch.Tensor,
                 ent2row: dict, rel2base: dict, reciprocal: bool,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.ent_emb = ent_emb.to(self.device)          # [n_ent, 2d]
        self.rel_emb = rel_emb.to(self.device)          # [n_rel or 2*n_rel, 2d]
        self.ent2row = ent2row
        self.rel2base = rel2base
        self.n_rel_base = len(rel2base)
        self.reciprocal = reciprocal
        self.dim = ent_emb.shape[1] // 2

    # -- construction -----------------------------------------------------

    @classmethod
    def from_libkge(cls, ckpt_path: str | Path, dataset_dir: str | Path,
                    device: str = "cpu") -> "ComplExScorer":
        """`dataset_dir` must hold the LibKGE archive's train/valid/test.txt
        (the exact files the ids were assigned from)."""
        ckpt = _load_raw_checkpoint(ckpt_path)
        state = ckpt["model"][0]
        # reciprocal checkpoints carry the wrapped copy under _base_model.*;
        # top-level and _base_model tensors are the same parameters.
        ent_emb = state["_entity_embedder.embeddings.weight"]
        rel_emb = state["_relation_embedder.embeddings.weight"]

        dataset_dir = Path(dataset_dir)
        train = dataset_dir / "train.txt"
        valid = dataset_dir / "valid.txt"
        test = dataset_dir / "test.txt"

        ent, rel = _first_appearance_maps([train])
        if len(ent) != ent_emb.shape[0]:
            # wnrr-style: ids were assigned over all three splits
            ent, rel = _first_appearance_maps([train, valid, test])
        if len(ent) != ent_emb.shape[0]:
            raise ValueError(
                f"entity map size {len(ent)} does not match checkpoint rows "
                f"{ent_emb.shape[0]} for {ckpt_path} -- wrong dataset_dir?")

        if rel_emb.shape[0] == 2 * len(rel):
            reciprocal = True
        elif rel_emb.shape[0] == len(rel):
            reciprocal = False
        else:
            raise ValueError(
                f"relation map size {len(rel)} incompatible with checkpoint "
                f"relation rows {rel_emb.shape[0]}")

        return cls(ent_emb, rel_emb, ent, rel, reciprocal, device=device)

    # -- complex helpers ---------------------------------------------------

    def _split(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return emb[:, :self.dim], emb[:, self.dim:]

    def _tail_query_vec(self, s_rows: torch.Tensor, r_rows: torch.Tensor) -> torch.Tensor:
        """u = s (x) p, laid out [re | im]; score over o = u @ ent_emb.T"""
        s_re, s_im = self._split(self.ent_emb[s_rows])
        p_re, p_im = self._split(self.rel_emb[r_rows])
        u_re = s_re * p_re - s_im * p_im
        u_im = s_re * p_im + s_im * p_re
        return torch.cat([u_re, u_im], dim=1)

    def _head_query_vec(self, r_rows: torch.Tensor, o_rows: torch.Tensor) -> torch.Tensor:
        """plain ComplEx head query: score(s) = v . [s_re | s_im] with
        v_re = p_re*o_re + p_im*o_im ; v_im = p_re*o_im - p_im*o_re"""
        p_re, p_im = self._split(self.rel_emb[r_rows])
        o_re, o_im = self._split(self.ent_emb[o_rows])
        v_re = p_re * o_re + p_im * o_im
        v_im = p_re * o_im - p_im * o_re
        return torch.cat([v_re, v_im], dim=1)

    # -- public scoring (row-indexed; see translate helpers below) ---------

    def score_tails_all(self, s_rows: torch.Tensor, r_rows: torch.Tensor) -> torch.Tensor:
        """[B] queries -> [B, n_ent] scores over every candidate tail."""
        u = self._tail_query_vec(s_rows.to(self.device), r_rows.to(self.device))
        return u @ self.ent_emb.T

    def score_heads_all(self, r_rows: torch.Tensor, o_rows: torch.Tensor) -> torch.Tensor:
        """[B] queries -> [B, n_ent] scores over every candidate head."""
        r_rows = r_rows.to(self.device)
        o_rows = o_rows.to(self.device)
        if self.reciprocal:
            u = self._tail_query_vec(o_rows, r_rows + self.n_rel_base)
        else:
            u = self._head_query_vec(r_rows, o_rows)
        return u @ self.ent_emb.T

    def score_hrt(self, h_rows: torch.Tensor, r_rows: torch.Tensor,
                  t_rows: torch.Tensor) -> torch.Tensor:
        """[B] pointwise scores for (h, r, t) triples."""
        u = self._tail_query_vec(h_rows.to(self.device), r_rows.to(self.device))
        return (u * self.ent_emb[t_rows.to(self.device)]).sum(dim=1)

    # -- string translation -------------------------------------------------

    def ent_rows(self, names: list[str]) -> torch.Tensor:
        return torch.tensor([self.ent2row[n] for n in names], dtype=torch.long)

    def rel_rows(self, names: list[str]) -> torch.Tensor:
        return torch.tensor([self.rel2base[n] for n in names], dtype=torch.long)

    # -- the correctness gate ------------------------------------------------

    @staticmethod
    def _read_triples(path: Path) -> list[tuple[str, str, str]]:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    out.append(tuple(parts))
        return out

    def filtered_mrr(self, dataset_dir: str | Path, split: str = "test",
                     batch: int = 512) -> dict:
        """Filtered MRR/Hits@10 on `split`, filtering against all three splits.

        Must reproduce the published LibKGE numbers (FB15K-237 ComplEx test
        MRR 0.348, WN18RR 0.475) -- this single number proves the unpickling,
        the re|im layout, the reciprocal handling and the id maps all at once.
        """
        dataset_dir = Path(dataset_dir)
        splits = {s: self._read_triples(dataset_dir / f"{s}.txt")
                  for s in ("train", "valid", "test")}
        n_ent = self.ent_emb.shape[0]

        # filter sets: true tails per (h_row, r_row), true heads per (r_row, t_row)
        true_tails: dict[tuple[int, int], list[int]] = {}
        true_heads: dict[tuple[int, int], list[int]] = {}
        for triples in splits.values():
            for h, r, t in triples:
                if h not in self.ent2row or t not in self.ent2row or r not in self.rel2base:
                    continue
                hr, rr, tr = self.ent2row[h], self.rel2base[r], self.ent2row[t]
                true_tails.setdefault((hr, rr), []).append(tr)
                true_heads.setdefault((rr, tr), []).append(hr)

        eval_triples = [(self.ent2row[h], self.rel2base[r], self.ent2row[t])
                        for h, r, t in splits[split]
                        if h in self.ent2row and t in self.ent2row and r in self.rel2base]
        skipped = len(splits[split]) - len(eval_triples)

        rr_sum, hits10, n_q = 0.0, 0, 0
        with torch.no_grad():
            for lo in range(0, len(eval_triples), batch):
                chunk = eval_triples[lo:lo + batch]
                hs = torch.tensor([c[0] for c in chunk])
                rs = torch.tensor([c[1] for c in chunk])
                ts = torch.tensor([c[2] for c in chunk])
                for direction in ("tail", "head"):
                    if direction == "tail":
                        scores = self.score_tails_all(hs, rs)
                        targets, keys, filt = ts, list(zip(hs.tolist(), rs.tolist())), true_tails
                    else:
                        scores = self.score_heads_all(rs, ts)
                        targets, keys, filt = hs, list(zip(rs.tolist(), ts.tolist())), true_heads
                    gold = scores[torch.arange(len(chunk)), targets].clone()
                    for i, key in enumerate(keys):
                        rows = filt.get(key)
                        if rows:
                            scores[i, rows] = float("-inf")
                    ranks = (scores > gold.unsqueeze(1)).sum(dim=1) + 1
                    rr_sum += (1.0 / ranks.float()).sum().item()
                    hits10 += (ranks <= 10).sum().item()
                    n_q += len(chunk)

        return {"split": split, "n_queries": n_q, "skipped_triples": skipped,
                "mrr": rr_sum / n_q, "hits@10": hits10 / n_q,
                "n_ent": n_ent, "reciprocal": self.reciprocal}


def main() -> int:  # pragma: no cover - thin CLI, exercised manually
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the filtered-MRR loader gate.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True,
                    help="LibKGE archive dir holding train/valid/test.txt")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--expected_mrr", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--report", default=None, help="optional JSON report path")
    args = ap.parse_args()

    scorer = ComplExScorer.from_libkge(args.ckpt, args.dataset_dir)
    print(f"loaded: n_ent={scorer.ent_emb.shape[0]} dim=2x{scorer.dim} "
          f"n_rel={scorer.n_rel_base} reciprocal={scorer.reciprocal}")
    res = scorer.filtered_mrr(args.dataset_dir, split=args.split)
    print(json.dumps(res, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if args.expected_mrr is not None:
        ok = abs(res["mrr"] - args.expected_mrr) <= args.tolerance
        print(f"GATE {'PASS' if ok else 'FAIL'}: mrr={res['mrr']:.4f} "
              f"expected={args.expected_mrr}±{args.tolerance}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
