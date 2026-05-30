"""Inference playground: load a trained GAN and watch it generate negatives.

Run the whole file end-to-end:
    python experiments/gan/playground_test.py

By default it loads the small checkpoint produced by `playground_train.py`.
Pass --ckpt to use a different one (e.g. the production `dummy.pt`).

What you'll see:
  1. Load a checkpoint, inspect what's in it
  2. Pick one real triple, run G once, look at the raw logits
  3. Pick a slot (head/rel/tail), mask the original, decode a new value
  4. Check whether the candidate collides with the real graph
  5. The retry loop in action when a collision happens
  6. Batch inference using the same code ADKGD calls at training time
  7. Qualitative inspection: do the negatives look "plausible-but-wrong"?
"""
# %%
import argparse
import os
import sys

import numpy as np
import torch

# Make sibling modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import load_checkpoint, generate_negatives, render_stats  # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "outputs", "checkpoints", "playground.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="Path to a checkpoint produced by train.py or playground_train.py")
args, _ = parser.parse_known_args()


# %%
# ----------------------------------------------------------------------
# 1. Load the checkpoint and look at what's inside
# ----------------------------------------------------------------------
print("=" * 70)
print("1) Load the checkpoint")
print("=" * 70)

if not os.path.exists(args.ckpt):
    print(f"\n  ERROR: {args.ckpt} not found.")
    print(f"  Run 'python experiments/gan/playground_train.py' first to produce it,")
    print(f"  or pass --ckpt experiments/gan/outputs/checkpoints/dummy.pt to use the production checkpoint.")
    sys.exit(1)

payload = load_checkpoint(args.ckpt, device=torch.device("cpu"))

print(f"\n  Checkpoint: {args.ckpt}")
print(f"  Top-level keys in payload:")
for k in sorted(payload.keys()):
    v = payload[k]
    if isinstance(v, dict):
        print(f"    {k:<18} (dict, {len(v)} entries)")
    elif isinstance(v, set):
        print(f"    {k:<18} (set,  {len(v):,} entries)")
    elif hasattr(v, "shape"):
        print(f"    {k:<18} (tensor shape {tuple(v.shape)})")
    else:
        print(f"    {k:<18} = {v}")

n_ent = payload["n_ent"]
n_rel = payload["n_rel"]
dim = payload["generator"].dim
z_dim = payload["z_dim"]
print(f"\n  Quick model facts:")
print(f"    embedding dim   : {dim}")
print(f"    noise vector dim: {z_dim}")
print(f"    entity vocab    : {n_ent}")
print(f"    relation vocab  : {n_rel}")
print(f"    real triples in graph (for collision check): {len(payload['real_triple_set']):,}")


# %%
# ----------------------------------------------------------------------
# 2. Pick one real triple, run the generator, look at raw logits
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) Generator forward on ONE triple")
print("=" * 70)

G = payload["generator"]
id2ent = payload["id2ent"]
id2rel = payload["id2rel"]

# Pick a real triple - choose one we can print and reason about.
# The first triple in the dummy_kg vocab is usually a meaningful one.
sample_h_id, sample_r_id, sample_t_id = list(payload["real_triple_set"])[0]
print(f"\n  Real triple chosen:")
print(f"    ({id2ent[sample_h_id]}, {id2rel[sample_r_id]}, {id2ent[sample_t_id]})")
print(f"    IDs: h={sample_h_id}, r={sample_r_id}, t={sample_t_id}")

h_t = torch.tensor([sample_h_id])
r_t = torch.tensor([sample_r_id])
t_t = torch.tensor([sample_t_id])
z = torch.randn(1, z_dim)
print(f"\n  Noise vector z (shape {tuple(z.shape)}): {z[0].numpy().round(3)}")

with torch.no_grad():
    head_logits, rel_logits, tail_logits = G(h_t, r_t, t_t, z)

print(f"\n  Raw logits from G:")
print(f"    head_logits shape = {tuple(head_logits.shape)}  ({n_ent} scores, one per entity)")
print(f"    rel_logits  shape = {tuple(rel_logits.shape)}   ({n_rel} scores, one per relation)")
print(f"    tail_logits shape = {tuple(tail_logits.shape)}")

# Show the top-3 picks G would make per slot.
def top3(logits, vocab_map):
    sorted_indices = logits[0].argsort(descending=True)[:3]
    return [(idx.item(), vocab_map[idx.item()], logits[0, idx].item()) for idx in sorted_indices]

print(f"\n  Top-3 candidate HEADS by logit score:")
for idx, name, score in top3(head_logits, id2ent):
    marker = "  <-- original" if idx == sample_h_id else ""
    print(f"    {score:>+7.3f}    id={idx}    {name}{marker}")

print(f"\n  Top-3 candidate RELATIONS by logit score:")
for idx, name, score in top3(rel_logits, id2rel):
    marker = "  <-- original" if idx == sample_r_id else ""
    print(f"    {score:>+7.3f}    id={idx}    {name}{marker}")

print(f"\n  Top-3 candidate TAILS by logit score:")
for idx, name, score in top3(tail_logits, id2ent):
    marker = "  <-- original" if idx == sample_t_id else ""
    print(f"    {score:>+7.3f}    id={idx}    {name}{marker}")


# %%
# ----------------------------------------------------------------------
# 3. Pick a slot, mask the original, decode a replacement
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) Slot picking + masked argmax")
print("=" * 70)

# In production, the slot is picked uniformly at random per positive.
# Here we'll demo all three slots for the same triple, deterministically.
for slot_idx, slot_name in enumerate(("head", "relation", "tail")):
    print(f"\n  --- if we corrupt the {slot_name.upper()} slot ---")
    if slot_idx == 0:
        logits = head_logits[0].clone()
        clean_value = sample_h_id
        vocab_map = id2ent
    elif slot_idx == 1:
        logits = rel_logits[0].clone()
        clean_value = sample_r_id
        vocab_map = id2rel
    else:
        logits = tail_logits[0].clone()
        clean_value = sample_t_id
        vocab_map = id2ent

    print(f"    original index  = {clean_value}  ('{vocab_map[clean_value]}')")
    print(f"    logits[original] before masking = {logits[clean_value].item():+.3f}")

    # Mask the original so argmax can't pick it.
    logits[clean_value] = float("-inf")
    print(f"    logits[original] after masking  = -inf    (forces the slot to move)")

    # Add Gumbel noise + argmax. The noise gives variety across calls.
    u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    new_idx = int((logits + gumbel * 0.5).argmax().item())

    print(f"    new index after argmax(logits + gumbel) = {new_idx}  ('{vocab_map[new_idx]}')")

    if slot_idx == 0:
        candidate = (new_idx, sample_r_id, sample_t_id)
    elif slot_idx == 1:
        candidate = (sample_h_id, new_idx, sample_t_id)
    else:
        candidate = (sample_h_id, sample_r_id, new_idx)
    print(f"    candidate triple: ({id2ent[candidate[0]]}, {id2rel[candidate[1]]}, {id2ent[candidate[2]]})")
    print(f"    collides with real graph? {candidate in payload['real_triple_set']}")


# %%
# ----------------------------------------------------------------------
# 4. The retry loop: what happens on collision
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) Retry loop demo")
print("=" * 70)
print("""
  When the GAN suggests a candidate that's already in the real graph, we
  re-roll. Each retry adds fresh Gumbel noise, so the argmax can drift to
  a different (hopefully still plausible) index.

  Stopping rules:
    - Found a candidate not in the real graph AND not a self-loop: keep it.
    - Hit max_retries: fall back to uniform-random replacement.
""")

# Pick the slot most likely to collide on dummy_kg (small relation vocab).
slot_idx = 1  # relation
clean_value = sample_r_id
print(f"  Corrupting relation slot of ({id2ent[sample_h_id]}, *, {id2ent[sample_t_id]})")
print(f"  Original relation: '{id2rel[clean_value]}'")

base_logits = rel_logits[0].clone()
base_logits[clean_value] = float("-inf")

max_retries = 5
for attempt in range(1, max_retries + 1):
    u = torch.rand_like(base_logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    new_idx = int((base_logits + gumbel * 0.5).argmax().item())
    candidate = (sample_h_id, new_idx, sample_t_id)
    collides = candidate in payload["real_triple_set"]
    print(f"    attempt {attempt}: relation -> '{id2rel[new_idx]}'    "
          f"{'COLLISION -> retry' if collides else 'accepted OK'}")
    if not collides:
        break


# %%
# ----------------------------------------------------------------------
# 5. Batch inference - the exact code ADKGD calls
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5) Batch inference (the production path)")
print("=" * 70)

# In production, ADKGD passes its OWN integer IDs (different from the GAN's),
# and the round-trip through strings keeps both worlds aligned. Here, since
# we're "playing both roles", we'll just use the GAN's IDs as if they were
# ADKGD's, and pass identity maps as the round-trip.
adkgd_id2ent = payload["id2ent"]
adkgd_id2rel = payload["id2rel"]
adkgd_ent2id = payload["ent2id"]
adkgd_rel2id = payload["rel2id"]

# Pick a small batch of real triples to corrupt.
sample_triples = list(payload["real_triple_set"])[:5]
print(f"\n  Input batch ({len(sample_triples)} real triples):")
for i, (h, r, t) in enumerate(sample_triples):
    print(f"    [{i}] ({adkgd_id2ent[h]}, {adkgd_id2rel[r]}, {adkgd_id2ent[t]})")

adkgd_maps = {
    "id2ent": adkgd_id2ent, "id2rel": adkgd_id2rel,
    "ent2id": adkgd_ent2id, "rel2id": adkgd_rel2id,
}
rng = np.random.default_rng(0)
negs, stats = generate_negatives(sample_triples, payload, adkgd_maps, rng=rng)

print(f"\n  Output batch (one negative per real triple):")
for i, ((rh, rr, rt), (nh, nr, nt)) in enumerate(zip(sample_triples, negs)):
    # Highlight which slot moved.
    moved = []
    if rh != nh:
        moved.append("head")
    if rr != nr:
        moved.append("rel")
    if rt != nt:
        moved.append("tail")
    print(f"    [{i}] ({adkgd_id2ent[nh]}, {adkgd_id2rel[nr]}, {adkgd_id2ent[nt]})    moved: {moved}")

print(f"\n  Stats: {render_stats(stats)}")
print(f"\n  Notes:")
print(f"    - 'processed' = batch size")
print(f"    - 'retries' counts re-rolls inside the slot decode (high = many collisions)")
print(f"    - 'uniform_fallbacks' counts items that exhausted retries (should usually be 0)")
print(f"    - Slot distribution should be near 1/3 each over a large batch")


# %%
# ----------------------------------------------------------------------
# 6. Qualitative check: do the negatives look "plausible-but-wrong"?
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6) Qualitative check on a larger batch")
print("=" * 70)

bigger_batch = list(payload["real_triple_set"])  # all real triples once
rng = np.random.default_rng(42)
negs, stats = generate_negatives(bigger_batch, payload, adkgd_maps, rng=rng)

print(f"\n  Ran on {len(bigger_batch)} real triples. Stats:")
print(f"    {render_stats(stats)}")

# Sample some random rows to eyeball.
print(f"\n  10 random (real -> generated negative) examples:")
sample_idx = rng.choice(len(bigger_batch), size=min(10, len(bigger_batch)), replace=False)
for i in sample_idx:
    rh, rr, rt = bigger_batch[i]
    nh, nr, nt = negs[i]
    print(f"    ({adkgd_id2ent[rh]:<22}, {adkgd_id2rel[rr]:<18}, {adkgd_id2ent[rt]:<22})")
    print(f"      ->")
    print(f"    ({adkgd_id2ent[nh]:<22}, {adkgd_id2rel[nr]:<18}, {adkgd_id2ent[nt]:<22})\n")

print("  Reading the output:")
print("    - In each pair, the negative should differ from the real triple at exactly ONE slot.")
print("    - On dummy_kg the vocab is tiny (10 entities, 3 relations) so negatives may not look")
print("      particularly 'plausible' - there isn't much room to be subtle.")
print("    - On FB15K-237 (14k entities, 237 relations), a well-trained GAN should swap")
print("      'Australia' for 'Japan' (both countries), not for 'Frank' (a person).")
print()
print("  Done.")
