"""Adapter that calls kggan in-process to produce one negative per positive.

OUR file (not kggan's). Lives next to kggan's `src/` so that the sys.path
setup below makes both kggan's bare imports (`from kg_data.loader import ...`)
and the pickled `TrainConfig`'s module path (`training.train_triple_gan`)
resolve from one location.

Consumer: ADKGD's `dataset.py:Reader._gan_negatives()`. It hands us a list of
ADKGD-ID (h, r, t) tuples; we hand back a list of ADKGD-ID negatives, having
round-tripped through strings → kggan IDs → masked-decode → kggan IDs → strings
→ ADKGD IDs.

The negative-generation logic (masked decode + collision retry + uniform
fallback) is reused directly from `sampling.hashmap_export`; we only refactor
the I/O so it returns the result instead of writing a TSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make kggan's package root importable. This single insertion covers both:
#   1. kggan's bare top-level imports (`from kg_data.loader import KnowledgeGraph`)
#   2. the pickled TrainConfig in dummy.pt (saved against module path
#      'training.train_triple_gan')
_KGGAN_SRC = Path(__file__).resolve().parent / "src"
if str(_KGGAN_SRC) not in sys.path:
    sys.path.insert(0, str(_KGGAN_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kg_data.loader import KnowledgeGraph, load_kg_union  # noqa: E402
from models.triple_gan import build_triple_embedding  # noqa: E402
from sampling.hashmap_export import (  # noqa: E402
    _argmax_with_gumbel_and_input_mask,
    _uniform_random_fallback,
    load_checkpoint,
)

# Mirror the slot constants from hashmap_export rather than importing the
# underscore-prefixed names (keeps lint clean).
_SLOT_HEAD = 0
_SLOT_REL = 1
_SLOT_TAIL = 2


def load_kggan(checkpoint_path, *, device=None):
    """Load the generator+embeddings once and return a payload dict.

    Caller stashes this on `self` and passes it to `generate_negatives` per
    training batch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(checkpoint_path, device=device)
    payload["device"] = device
    return payload


def build_kg(dataset_dir):
    """Construct the KnowledgeGraph kggan expects from ADKGD's data files.

    Uses `load_kg_union` because ADKGD operates on the train+valid+test union;
    kggan's collision check (`kg.triple_set_idx`) must therefore see the same
    union so it never emits a real triple as a negative.
    """
    return load_kg_union(dataset_dir)


def generate_negatives(
    pos_triples_adkgd,
    *,
    payload,
    kg,
    adkgd_id2ent,
    adkgd_id2rel,
    adkgd_ent2id,
    adkgd_rel2id,
    rng=None,
    gumbel_temperature=0.5,
    batch_size=256,
    max_retries=20,
):
    """For each ADKGD-ID positive, generate one ADKGD-ID negative via kggan.

    Returns (negatives_list, stats_dict). `negatives_list[i]` is the negative
    paired with `pos_triples_adkgd[i]`.

    Treats real positives and injected anomalies uniformly — kggan runs a
    forward pass for every input.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    generator = payload["generator"]
    entity_embedding = payload["entity_embedding"]
    relation_embedding = payload["relation_embedding"]
    device = payload["device"]

    generator.eval()
    entity_embedding.eval()
    relation_embedding.eval()

    num_entities = generator.num_entities
    num_relations = generator.num_relations
    real_triple_set = kg.triple_set_idx  # in kggan ID space

    row_to_entity_id = {row: ident for ident, row in kg.entity_id_to_row.items()}
    channel_to_relation = {ch: ident for ident, ch in kg.relation_id_to_channel.items()}

    # ADKGD IDs → strings → kggan IDs. Vocab is the union of train+valid+test
    # on both sides, so every string should resolve.
    kggan_ids = []
    for h_a, r_a, t_a in pos_triples_adkgd:
        h_s = adkgd_id2ent[h_a]
        r_s = adkgd_id2rel[r_a]
        t_s = adkgd_id2ent[t_a]
        kggan_ids.append((
            kg.entity_id_to_row[h_s],
            kg.relation_id_to_channel[r_s],
            kg.entity_id_to_row[t_s],
        ))

    out = []
    stats = {
        "processed": 0,
        "retries": 0,
        "uniform_fallbacks": 0,
        "slot_head": 0,
        "slot_rel": 0,
        "slot_tail": 0,
    }

    for batch_start in range(0, len(kggan_ids), batch_size):
        batch = kggan_ids[batch_start:batch_start + batch_size]
        batch_array = np.asarray(batch, dtype=np.int64)
        clean_batch_tensor = torch.from_numpy(batch_array).to(device)
        current_batch_size = clean_batch_tensor.size(0)

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

        slots_picked = rng.integers(3, size=current_batch_size)
        clean_h_list = clean_batch_tensor[:, 0].tolist()
        clean_r_list = clean_batch_tensor[:, 1].tolist()
        clean_t_list = clean_batch_tensor[:, 2].tolist()

        decoded_head = _argmax_with_gumbel_and_input_mask(
            gen_output["head_entity_logits"], clean_batch_tensor[:, 0], gumbel_temperature,
        ).tolist()
        decoded_rel = _argmax_with_gumbel_and_input_mask(
            gen_output["relation_logits"], clean_batch_tensor[:, 1], gumbel_temperature,
        ).tolist()
        decoded_tail = _argmax_with_gumbel_and_input_mask(
            gen_output["tail_entity_logits"], clean_batch_tensor[:, 2], gumbel_temperature,
        ).tolist()

        for i in range(current_batch_size):
            clean_h = clean_h_list[i]
            clean_r = clean_r_list[i]
            clean_t = clean_t_list[i]
            slot = int(slots_picked[i])

            if slot == _SLOT_HEAD:
                neg_h, neg_r, neg_t = decoded_head[i], clean_r, clean_t
            elif slot == _SLOT_REL:
                neg_h, neg_r, neg_t = clean_h, decoded_rel[i], clean_t
            else:
                neg_h, neg_r, neg_t = clean_h, clean_r, decoded_tail[i]

            # Retry same slot with fresh Gumbel noise on collision / self-loop.
            target_logits = (
                gen_output["head_entity_logits"] if slot == _SLOT_HEAD
                else gen_output["relation_logits"] if slot == _SLOT_REL
                else gen_output["tail_entity_logits"]
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
                sub_logits = target_logits[i:i + 1]
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
            stats["retries"] += retry_count

            # Uniform random fallback if the GAN keeps colliding.
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
                    stats["uniform_fallbacks"] += 1
                    neg_h, neg_r, neg_t = fallback

            if slot == _SLOT_HEAD:
                stats["slot_head"] += 1
            elif slot == _SLOT_REL:
                stats["slot_rel"] += 1
            else:
                stats["slot_tail"] += 1

            # kggan IDs → strings → ADKGD IDs.
            out.append((
                adkgd_ent2id[row_to_entity_id[neg_h]],
                adkgd_rel2id[channel_to_relation[neg_r]],
                adkgd_ent2id[row_to_entity_id[neg_t]],
            ))
        stats["processed"] += current_batch_size

    return out, stats


def render_stats(stats):
    total_slot = stats["slot_head"] + stats["slot_rel"] + stats["slot_tail"]
    if total_slot:
        slot_pct = (
            f"head={stats['slot_head']}/{total_slot}({stats['slot_head']/total_slot:.1%}) "
            f"rel={stats['slot_rel']}/{total_slot}({stats['slot_rel']/total_slot:.1%}) "
            f"tail={stats['slot_tail']}/{total_slot}({stats['slot_tail']/total_slot:.1%})"
        )
    else:
        slot_pct = "no slots"
    return (
        f"processed={stats['processed']:,}  "
        f"retries={stats['retries']:,}  "
        f"uniform_fallbacks={stats['uniform_fallbacks']:,}  "
        f"slot_distribution: {slot_pct}"
    )
