"""Pre-flight validation of a generated `gan_hashmap.tsv` before shipping.

This is the gate between kggan's exporter and ADKGD's consumer. If any
of the checks fails, the file is NOT safe to ship to ADKGD: either the
on-disk contract was violated (wrong column count, etc.) or the content
is wrong (collides with the real graph, misses real triples, etc.). The
script is loud on failure and exits non-zero so the SLURM job can abort
cleanly.

The six-column format is documented in
[src/sampling/hashmap_export.py](../sampling/hashmap_export.py). The
checks are:

  1. Parse:    every line has exactly 6 tab-separated columns.
  2. Vocab:    every string (orig and neg) is in kg's union vocab.
  3. Exactly-one-position-changed: the negative differs from the
     original at exactly one of the three triple positions (head, rel,
     or tail). Catches off-by-one bugs and pure-identity copies.
  4. Real-graph collision: zero negatives sit in `kg.triple_set_idx`.
  5. Per-positive uniqueness + completeness: every real triple in
     `kg.triple_set_idx` appears as `(orig_h, orig_r, orig_t)` exactly
     once. No missing entries; no duplicates.
  6. Coverage: row_count == len(kg.triple_set_idx).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from kg_data.loader import KnowledgeGraph


# Per-row slot recovery: which position changed.
_SLOT_LABELS = ("head", "rel", "tail")


@dataclass
class PreflightHashmapReport:
    """Result of `validate_hashmap_tsv`. `is_ok` aggregates all checks."""
    tsv_path:                       str
    rows_parsed:                    int = 0
    parse_failures:                 int = 0
    unknown_entity_rows:            int = 0
    unknown_relation_rows:          int = 0
    identity_rows:                  int = 0   # negative equals original at all 3 positions
    multi_position_change_rows:     int = 0   # >1 position differs (should be 0)
    real_graph_collisions:          int = 0
    expected_row_count:             int = 0
    triples_seen_once:              int = 0
    triples_missing:                int = 0
    triples_with_duplicates:        int = 0
    triples_not_in_real_graph:      int = 0   # orig triple wasn't in kg.triple_set_idx

    # Recovered slot distribution (sanity check on the exporter's RNG).
    slots_head:                     int = 0
    slots_rel:                      int = 0
    slots_tail:                     int = 0

    sample_rows:                    List[str] = field(default_factory=list)
    is_ok:                          bool = False
    failure_reasons:                List[str] = field(default_factory=list)

    def render(self) -> str:
        lines: List[str] = []
        lines.append(f"Hashmap pre-flight report for: {self.tsv_path}")
        lines.append(f"  Rows parsed:                          {self.rows_parsed:,}")
        lines.append(f"  Expected rows (unique real triples):  {self.expected_row_count:,}")
        lines.append(f"  Parse failures (wrong column count):  {self.parse_failures:,}")
        lines.append(f"  Unknown entity (orig or neg):         {self.unknown_entity_rows:,}")
        lines.append(f"  Unknown relation (orig or neg):       {self.unknown_relation_rows:,}")
        lines.append(f"  Identity rows (no position moved):    {self.identity_rows:,}")
        lines.append(f"  Multi-position-change rows:           {self.multi_position_change_rows:,}")
        lines.append(f"  Real-graph collisions (neg in real):  {self.real_graph_collisions:,}")
        lines.append(f"  Triples seen exactly once:            {self.triples_seen_once:,}")
        lines.append(f"  Triples missing:                      {self.triples_missing:,}")
        lines.append(f"  Triples with duplicate entries:       {self.triples_with_duplicates:,}")
        lines.append(f"  Triples (orig) not in real graph:     {self.triples_not_in_real_graph:,}")
        total_slots = self.slots_head + self.slots_rel + self.slots_tail
        if total_slots > 0:
            lines.append(
                f"  Slot distribution:                    "
                f"head={self.slots_head}({self.slots_head/total_slots:.2%}) "
                f"rel={self.slots_rel}({self.slots_rel/total_slots:.2%}) "
                f"tail={self.slots_tail}({self.slots_tail/total_slots:.2%})"
            )
        if self.sample_rows:
            lines.append("  Sample (first 6 rows, eyeball):")
            for sample in self.sample_rows:
                lines.append(f"    {sample}")
        if self.is_ok:
            lines.append("Verdict: OK to use.")
        else:
            lines.append("Verdict: NOT OK - " + "; ".join(self.failure_reasons))
        return "\n".join(lines)


def validate_hashmap_tsv(
    tsv_path: str | Path,
    kg: KnowledgeGraph,
    *,
    sample_size: int = 6,
) -> PreflightHashmapReport:
    """Run the six hashmap checks against `tsv_path`."""
    tsv_path = Path(tsv_path)
    # `kg.triple_set_idx` is the deduped union; the hashmap writes one
    # row per unique real triple, so the expected count uses the set
    # length not the list length (the latter may include duplicates if a
    # triple appears in multiple splits).
    report = PreflightHashmapReport(
        tsv_path=str(tsv_path),
        expected_row_count=len(kg.triple_set_idx),
    )

    if not tsv_path.exists():
        report.failure_reasons.append(f"file does not exist: {tsv_path}")
        return report

    entity_id_to_row       = kg.entity_id_to_row
    relation_id_to_channel = kg.relation_id_to_channel
    real_triple_set        = kg.triple_set_idx

    # Track how many times each orig triple appears.
    counts_per_orig: Dict[Tuple[str, str, str], int] = defaultdict(int)

    with tsv_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            stripped = raw_line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue
            report.rows_parsed += 1
            if len(report.sample_rows) < sample_size:
                report.sample_rows.append(stripped)
            columns = stripped.split("\t")
            if len(columns) != 6:
                report.parse_failures += 1
                continue
            orig_h, orig_r, orig_t, neg_h, neg_r, neg_t = columns

            # Check 2 — vocab
            head_known = (orig_h in entity_id_to_row) and (neg_h in entity_id_to_row)
            tail_known = (orig_t in entity_id_to_row) and (neg_t in entity_id_to_row)
            rel_known  = (orig_r in relation_id_to_channel) and (neg_r in relation_id_to_channel)
            if not (head_known and tail_known):
                report.unknown_entity_rows += 1
                continue
            if not rel_known:
                report.unknown_relation_rows += 1
                continue

            # Check 3 — exactly one position changed
            differs = (
                int(orig_h != neg_h),
                int(orig_r != neg_r),
                int(orig_t != neg_t),
            )
            num_changes = sum(differs)
            if num_changes == 0:
                report.identity_rows += 1
                # Still count below for other checks.
            elif num_changes > 1:
                report.multi_position_change_rows += 1
            else:
                # Recover which slot was moved for the distribution telemetry.
                slot_index = differs.index(1)
                if slot_index == 0:
                    report.slots_head += 1
                elif slot_index == 1:
                    report.slots_rel += 1
                else:
                    report.slots_tail += 1

            # Check 4 — real-graph collision on the NEGATIVE
            neg_idx = (
                entity_id_to_row[neg_h],
                relation_id_to_channel[neg_r],
                entity_id_to_row[neg_t],
            )
            if neg_idx in real_triple_set:
                report.real_graph_collisions += 1

            # Check 5 (part 1) — orig must be a real triple
            orig_idx = (
                entity_id_to_row[orig_h],
                relation_id_to_channel[orig_r],
                entity_id_to_row[orig_t],
            )
            if orig_idx not in real_triple_set:
                report.triples_not_in_real_graph += 1

            counts_per_orig[(orig_h, orig_r, orig_t)] += 1

    # Check 5 (part 2) — per-positive completeness across the deduped
    # union triples. Build the string-form set of real triples and
    # cross-reference.
    row_to_entity_id    = {v: k for k, v in entity_id_to_row.items()}
    channel_to_relation = {v: k for k, v in relation_id_to_channel.items()}
    real_triple_strs: Set[Tuple[str, str, str]] = set()
    for head_index, relation_index, tail_index in real_triple_set:
        real_triple_strs.add((
            row_to_entity_id[head_index],
            channel_to_relation[relation_index],
            row_to_entity_id[tail_index],
        ))

    for triple_str in real_triple_strs:
        count = counts_per_orig.get(triple_str, 0)
        if count == 0:
            report.triples_missing += 1
        elif count == 1:
            report.triples_seen_once += 1
        else:
            report.triples_with_duplicates += 1

    # Verdict
    if report.parse_failures > 0:
        report.failure_reasons.append(f"{report.parse_failures} lines failed to parse")
    if report.rows_parsed != report.expected_row_count:
        report.failure_reasons.append(
            f"row count {report.rows_parsed:,} != expected {report.expected_row_count:,}"
        )
    if report.unknown_entity_rows + report.unknown_relation_rows > 0:
        report.failure_reasons.append(
            f"{report.unknown_entity_rows + report.unknown_relation_rows} rows had unknown vocab"
        )
    if report.identity_rows > 0:
        report.failure_reasons.append(
            f"{report.identity_rows} rows had negative == original at all 3 positions"
        )
    if report.multi_position_change_rows > 0:
        report.failure_reasons.append(
            f"{report.multi_position_change_rows} rows changed more than one position"
        )
    if report.real_graph_collisions > 0:
        report.failure_reasons.append(
            f"{report.real_graph_collisions} negatives collided with the real graph"
        )
    if report.triples_missing > 0:
        report.failure_reasons.append(
            f"{report.triples_missing} real triples missing from the hashmap"
        )
    if report.triples_with_duplicates > 0:
        report.failure_reasons.append(
            f"{report.triples_with_duplicates} real triples had multiple entries"
        )
    report.is_ok = not report.failure_reasons
    return report
