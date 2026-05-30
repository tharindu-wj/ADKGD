"""CLI: train the TripleGAN on a KG and save a checkpoint.

Usage (dummy KG smoke):
    python scripts/train_gan.py --data data/dummy_kg \\
        --epochs 50 --batch-size 32 --device cpu \\
        --out outputs/checkpoints/dummy.pt

Usage (FB15k-237 on V100):
    python scripts/train_gan.py --data data/FB15K \\
        --epochs 400 --batch-size 256 --device cuda \\
        --out outputs/checkpoints/fb15k.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_data.loader import load_kg_union              # noqa: E402
from training.train_triple_gan import (               # noqa: E402
    TrainConfig,
    save_checkpoint,
    train_triple_gan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",   required=True, type=Path,
                        help="Dataset directory with train.txt / valid.txt / test.txt.")
    parser.add_argument("--out",    required=True, type=Path,
                        help="Final checkpoint path. Intermediate checkpoints (if "
                             "--checkpoint-every > 0) land in the same directory.")
    parser.add_argument("--epochs",     type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     type=str, default=None,
                        help="cuda or cpu. Default: cuda if available else cpu.")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--embedding-dim",       type=int, default=200)
    parser.add_argument("--latent-dim",          type=int, default=32)
    parser.add_argument("--bottleneck-hidden-dim", type=int, default=512)
    parser.add_argument("--training-rounds",     type=int, default=4000,
                        help="Number of TRIC corruption rounds when building pairs.")
    parser.add_argument("--corruption-steps-per-round", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save an intermediate checkpoint every N epochs. "
                             "0 = only the final checkpoint.")
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--w-recon",            type=float, default=20.0)
    parser.add_argument("--w-adv",              type=float, default=1.0)
    parser.add_argument("--w-div",              type=float, default=0.5)
    parser.add_argument("--restrict-tric-to-change-relation", action="store_true",
                        help="Restrict TRIC to change_relation only (the notebook's "
                             "dummy-KG setting). FB15k-237 default is the full mix.")
    args = parser.parse_args()

    if args.device is None:
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_string = args.device
    device = torch.device(device_string)
    print(f"[train_gan] device: {device}", flush=True)

    print(f"[train_gan] loading union KG from {args.data} ...", flush=True)
    kg = load_kg_union(args.data)
    print(
        f"[train_gan] entities={len(kg.entity_id_to_row):,} "
        f"relations={len(kg.relation_id_to_channel):,} "
        f"union_triples={len(kg.triples_idx):,}",
        flush=True,
    )

    has_types = any(t != "unknown" for t in kg.node_types.values())
    if not has_types:
        print(
            "[train_gan] WARNING: kg.node_types is empty/unknown - TRIC's "
            "swap_subject_same_type / swap_object_same_type ops will be disabled. "
            "Run scripts/build_pseudo_types.py first for FB15K-237.",
            flush=True,
        )

    tric_operation_weights = (
        {"change_relation": 1} if args.restrict_tric_to_change_relation else None
    )

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        bottleneck_hidden_dim=args.bottleneck_hidden_dim,
        num_corruption_steps_per_round=args.corruption_steps_per_round,
        num_training_rounds=args.training_rounds,
        tric_operation_weights=tric_operation_weights,
        total_epochs=args.epochs,
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_weight_reconstruction=args.w_recon,
        loss_weight_adversarial=args.w_adv,
        loss_weight_diversity=args.w_div,
        checkpoint_every_n_epochs=args.checkpoint_every,
        random_seed=args.seed,
    )

    trained = train_triple_gan(
        clean_triples=list(kg.triples_idx),
        num_entities=len(kg.entity_id_to_row),
        num_relations=len(kg.relation_id_to_channel),
        node_types=kg.node_types if has_types else None,
        config=config,
        device=device,
        checkpoint_dir=args.out.parent,
        checkpoint_prefix=args.out.stem,
    )

    save_checkpoint(
        args.out,
        generator=trained['generator'],
        discriminator=trained['discriminator'],
        entity_embedding=trained['entity_embedding'],
        relation_embedding=trained['relation_embedding'],
        config=trained['config'],
        epoch=config.total_epochs,
    )
    print(f"[train_gan] final checkpoint: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
