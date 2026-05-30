"""Export the kggan GAN as a 6-column hashmap TSV consumed by ADKGD.

The integration contract with ADKGD (Approach 2c — see plan file):

  * One row per UNIQUE real triple in `kg.triple_set_idx`.
  * 6 tab-separated columns:
      orig_h <TAB> orig_r <TAB> orig_t <TAB> neg_h <TAB> neg_r <TAB> neg_t
  * UTF-8, LF endings, no header, no trailing whitespace.
  * In each row, the negative triple equals the original at exactly two
    of the three positions; the third position is the GAN's pick.
    Which position moves is decided HERE at export time by a uniform
    random slot pick per positive (matches ADKGD's existing 1/3-each
    slot distribution); the slot identity itself is not stored on disk.

ADKGD's loader keys this map by the `(orig_h, orig_r, orig_t)` 3-tuple
of strings (Python tuples hash natively — no composite-string key
needed). For each positive in its training batch, ADKGD does:

    table = {}
    with open(path) as f:
        for line in f:
            h, r, t, nh, nr, nt = line.rstrip('\\n').split('\\t')
            table[(h, r, t)] = (nh, nr, nt)
    # ...
    neg = table.get((h_str, r_str, t_str))   # one negative per positive

Real triples missing from the map fall back to ADKGD's own random
corrupter.

The masked-single-position decode (clean index masked to -inf at the
chosen slot, Gumbel noise added, argmax over the remaining vocab) is
the same logic the old pool-based sampler used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from kg_data.loader import KnowledgeGraph
from models.triple_gan import (
    EntityEmbedding,
    RelationEmbedding,
    TripleGenerator,
    build_triple_embedding,
)


Triple = Tuple[int, int, int]

# Internal slot identifiers used inside this module. NOT written to disk.
_SLOT_HEAD = 0
_SLOT_REL  = 1
_SLOT_TAIL = 2


@dataclass
class ExportStats:
    """Summary returned by `export_hashmap_tsv` for the caller to log."""
    triples_processed:      int = 0
    rows_written:           int = 0
    fallbacks_uniform:      int = 0   # times the masked decode couldn't avoid a real-graph collision
    self_loops_uniform:     int = 0   # times a head=tail output was forced to be redrawn

    # Per-slot distribution of the chosen slot (sanity-check that the
    # uniform-random pick is roughly 1/3 each — if not, something's off
    # in the RNG plumbing).
    slot_counts_head:       int = 0
    slot_counts_rel:        int = 0
    slot_counts_tail:       int = 0

    def render(self) -> str:
        total_slot = self.slot_counts_head + self.slot_counts_rel + self.slot_counts_tail
        if total_slot > 0:
            slot_pct = (
                f"head={self.slot_counts_head}/{total_slot}({self.slot_counts_head/total_slot:.2%}) "
                f"rel={self.slot_counts_rel}/{total_slot}({self.slot_counts_rel/total_slot:.2%}) "
                f"tail={self.slot_counts_tail}/{total_slot}({self.slot_counts_tail/total_slot:.2%})"
            )
        else:
            slot_pct = "no slots picked"
        return (
            f"triples_processed={self.triples_processed:,}  "
            f"rows_written={self.rows_written:,}  "
            f"fallbacks_uniform={self.fallbacks_uniform:,}  "
            f"self_loops_uniform={self.self_loops_uniform:,}  "
            f"slot_distribution: {slot_pct}"
        )


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> Dict:
    """Reconstruct generator + embeddings from a checkpoint saved by
    `training.train_triple_gan.save_checkpoint`.
    """
    from training.train_triple_gan import TrainConfig  # local to avoid cycle

    payload: Dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: TrainConfig = payload['config']
    num_entities  = payload['num_entities']
    num_relations = payload['num_relations']

    entity_embedding   = EntityEmbedding(num_entities, config.embedding_dim).to(device)
    relation_embedding = RelationEmbedding(num_relations, config.embedding_dim).to(device)
    generator = TripleGenerator(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config.embedding_dim,
        latent_dim=config.latent_dim,
        encoder_base_channels=config.encoder_base_channels,
        encoder_num_conv_layers=config.encoder_num_conv_layers,
        encoder_num_attn_heads=config.encoder_num_attn_heads,
        bottleneck_hidden_dim=config.bottleneck_hidden_dim,
    ).to(device)

    generator.load_state_dict(payload['generator_state_dict'])
    entity_embedding.load_state_dict(payload['entity_embedding_state_dict'])
    relation_embedding.load_state_dict(payload['relation_embedding_state_dict'])

    return {
        'generator':          generator,
        'entity_embedding':   entity_embedding,
        'relation_embedding': relation_embedding,
        'config':             config,
        'epoch':              payload.get('epoch'),
    }


def _argmax_with_gumbel_and_input_mask(
    logits: torch.Tensor,
    clean_indices: torch.Tensor,
    gumbel_temperature: float,
) -> torch.Tensor:
    """Sample from `logits` with two adjustments:

      1. The clean index's logit is set to -inf, so argmax cannot return
         the input (guarantees the slot moves).
      2. Gumbel noise (scaled by `gumbel_temperature`) is added before
         argmax, so per-call output varies even when fed the same clean
         triple.
    """
    perturbed = logits.clone()
    perturbed.scatter_(1, clean_indices.unsqueeze(1), float('-inf'))
    if gumbel_temperature > 0:
        uniform_noise = torch.rand_like(perturbed).clamp_(1e-10, 1.0 - 1e-10)
        gumbel_noise  = -torch.log(-torch.log(uniform_noise))
        perturbed = perturbed + gumbel_noise * gumbel_temperature
    return perturbed.argmax(dim=-1)


def _uniform_random_fallback(
    *,
    clean_h: int,
    clean_r: int,
    clean_t: int,
    slot: int,
    num_entities: int,
    num_relations: int,
    real_triple_set: set,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> Tuple[int, int, int] | None:
    """Sample a uniform-random replacement at the given slot, rejecting
    self-loops and real-graph collisions. Used as a last-resort fallback
    when the GAN's masked decode keeps colliding.
    """
    target_max = num_relations if slot == _SLOT_REL else num_entities
    clean_value = clean_r if slot == _SLOT_REL else (clean_h if slot == _SLOT_HEAD else clean_t)
    for _ in range(max_attempts):
        candidate = int(rng.integers(target_max))
        if candidate == clean_value:
            continue
        if slot == _SLOT_HEAD:
            trial = (candidate, clean_r, clean_t)
        elif slot == _SLOT_REL:
            trial = (clean_h, candidate, clean_t)
        else:
            trial = (clean_h, clean_r, candidate)
        if trial[0] == trial[2]:
            continue
        if trial in real_triple_set:
            continue
        return trial
    return None


def export_hashmap_tsv(
    *,
    generator: TripleGenerator,
    entity_embedding: EntityEmbedding,
    relation_embedding: RelationEmbedding,
    kg: KnowledgeGraph,
    output_path: str | Path,
    gumbel_temperature: float = 0.5,
    batch_size: int = 256,
    max_retries: int = 20,
    device: torch.device | None = None,
    random_seed: int = 0,
) -> ExportStats:
    """Walk the UNIQUE real triples in `kg` and emit one row of the
    hashmap TSV per positive. Slot picked uniformly at random per
    positive; the GAN's masked-decode result for that slot fills the
    one moved position, the other two columns copy the input.

    Output rows are sorted by `(orig_h, orig_r, orig_t)` so runs with
    the same seed produce diff-stable files.
    """
    if device is None:
        device = next(generator.parameters()).device

    generator.eval()
    entity_embedding.eval()
    relation_embedding.eval()

    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    num_entities  = generator.num_entities
    num_relations = generator.num_relations

    # Dedupe + sort so the TSV is reproducible AND has no duplicate keys.
    # `kg.triples_idx` is the union over train/valid/test and may contain
    # the same triple in multiple splits; we want one row per unique triple.
    sorted_triples_idx: List[Triple] = sorted(set(kg.triples_idx))
    real_triple_set = kg.triple_set_idx

    # Inverse vocab maps for the string round-trip.
    row_to_entity_id    = {row: identifier for identifier, row in kg.entity_id_to_row.items()}
    channel_to_relation = {channel: identifier for identifier, channel in kg.relation_id_to_channel.items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for batch_start in range(0, len(sorted_triples_idx), batch_size):
            batch_end   = batch_start + batch_size
            batch_idx   = sorted_triples_idx[batch_start:batch_end]
            batch_array = np.asarray(batch_idx, dtype=np.int64)
            clean_batch_tensor = torch.from_numpy(batch_array).to(device)
            current_batch_size = clean_batch_tensor.size(0)

            # ── Forward pass: get logits over all three slots for the
            # whole batch. We only USE one slot's logits per item, but a
            # single forward pass is cheaper than three smaller ones.
            with torch.no_grad():
                clean_embedding = build_triple_embedding(
                    clean_batch_tensor, entity_embedding, relation_embedding,
                )
                latent_sample = torch.randn(
                    current_batch_size, generator.latent_dim, device=device,
                )
                gen_output = generator(
                    clean_embedding, latent_sample,
                    gumbel_temperature=gumbel_temperature,
                    return_soft_samples=False,
                )

            # ── Per-positive slot pick + decode.
            # Pick all slots up front so the rng calls happen in a
            # deterministic order regardless of any later branching.
            slots_picked = rng.integers(3, size=current_batch_size)

            clean_h_list = clean_batch_tensor[:, 0].tolist()
            clean_r_list = clean_batch_tensor[:, 1].tolist()
            clean_t_list = clean_batch_tensor[:, 2].tolist()

            # First: compute the masked-decode argmax for every position.
            # We always do all three because indexing is cheap and we
            # don't know per-item which one we'll use until we read
            # `slots_picked[i]`.
            decoded_head = _argmax_with_gumbel_and_input_mask(
                gen_output['head_entity_logits'],
                clean_batch_tensor[:, 0],
                gumbel_temperature,
            ).tolist()
            decoded_rel = _argmax_with_gumbel_and_input_mask(
                gen_output['relation_logits'],
                clean_batch_tensor[:, 1],
                gumbel_temperature,
            ).tolist()
            decoded_tail = _argmax_with_gumbel_and_input_mask(
                gen_output['tail_entity_logits'],
                clean_batch_tensor[:, 2],
                gumbel_temperature,
            ).tolist()

            # Now apply the per-item slot pick and finalize each row.
            for i in range(current_batch_size):
                clean_h = clean_h_list[i]
                clean_r = clean_r_list[i]
                clean_t = clean_t_list[i]
                slot    = int(slots_picked[i])

                # Build the candidate triple.
                if slot == _SLOT_HEAD:
                    neg_h, neg_r, neg_t = decoded_head[i], clean_r, clean_t
                elif slot == _SLOT_REL:
                    neg_h, neg_r, neg_t = clean_h, decoded_rel[i], clean_t
                else:  # _SLOT_TAIL
                    neg_h, neg_r, neg_t = clean_h, clean_r, decoded_tail[i]

                # Retry the SAME slot with fresh Gumbel noise on the same
                # logits if we hit a self-loop or real-graph collision.
                target_logits = (
                    gen_output['head_entity_logits'] if slot == _SLOT_HEAD
                    else gen_output['relation_logits'] if slot == _SLOT_REL
                    else gen_output['tail_entity_logits']
                )
                target_clean_value = (
                    clean_h if slot == _SLOT_HEAD
                    else clean_r if slot == _SLOT_REL
                    else clean_t
                )

                retry_count = 0
                while retry_count < max_retries and (
                    (neg_h == neg_t) or ((neg_h, neg_r, neg_t) in real_triple_set)
                ):
                    sub_logits = target_logits[i:i+1]
                    sub_clean = torch.tensor([target_clean_value], device=device)
                    redrawn = int(_argmax_with_gumbel_and_input_mask(
                        sub_logits, sub_clean, gumbel_temperature,
                    ).item())
                    if slot == _SLOT_HEAD:
                        neg_h = redrawn
                    elif slot == _SLOT_REL:
                        neg_r = redrawn
                    else:
                        neg_t = redrawn
                    retry_count += 1

                # If we exhausted retries, fall back to uniform random
                # for this single item.
                if (neg_h == neg_t) or ((neg_h, neg_r, neg_t) in real_triple_set):
                    fallback = _uniform_random_fallback(
                        clean_h=clean_h, clean_r=clean_r, clean_t=clean_t,
                        slot=slot,
                        num_entities=num_entities,
                        num_relations=num_relations,
                        real_triple_set=real_triple_set,
                        rng=rng,
                    )
                    if fallback is not None:
                        if neg_h == neg_t:
                            stats.self_loops_uniform += 1
                        else:
                            stats.fallbacks_uniform += 1
                        neg_h, neg_r, neg_t = fallback

                # Track which slot we used (sanity check on RNG).
                if slot == _SLOT_HEAD:
                    stats.slot_counts_head += 1
                elif slot == _SLOT_REL:
                    stats.slot_counts_rel += 1
                else:
                    stats.slot_counts_tail += 1

                # Write the row.
                output_file.write(
                    f"{row_to_entity_id[clean_h]}\t{channel_to_relation[clean_r]}\t{row_to_entity_id[clean_t]}\t"
                    f"{row_to_entity_id[neg_h]}\t{channel_to_relation[neg_r]}\t{row_to_entity_id[neg_t]}\n"
                )
                stats.rows_written += 1
            stats.triples_processed += current_batch_size

    return stats
