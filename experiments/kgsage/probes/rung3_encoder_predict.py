"""RUNG 3 - Is the encoder good at prediction?   (HPC: needs torch_geometric + encoder ckpt)

Question: given (h, r, ?), does the encoder rank the RIGHT tail near the top, and
are its other guesses plausible? This is the no-MRR way to read "good at prediction".
"""
import random
from _common import banner, hint, load_dataset


def main():
    banner("RUNG 3 - Is the encoder good at prediction?")
    kg, data_dir, enc, gan = load_dataset()

    if enc is None:
        print("\n[skip] No encoder checkpoint at experiments/kgsage/outputs/fb15k237_encoder.pt.")
        print("       Train it first (HPC): python -m kgsage.cli.train_encoder --dataset fb15k237")
        return 0
    try:
        import torch
        from kgsage.encoder.models import KGSAGELinkPredictor
    except Exception as e:
        print(f"\n[skip] cannot import the encoder ({type(e).__name__}: {e}).")
        print("       The RGCN needs torch_geometric (HPC-only here).")
        return 0

    model, ent2id, rel2id = KGSAGELinkPredictor.load_pretrained(enc, device="cpu")
    id2ent = {i: s for s, i in ent2id.items()}
    id2rel = {i: s for s, i in rel2id.items()}

    h, r, t = kg["triples_train"][0]                         # a known-true fact
    with torch.no_grad():
        ent_emb = model.encoder(kg["edge_index"], kg["edge_type"])     # run the RGCN once
        rel_vec = model.decoder.rel_emb.weight[r]
        scores = (ent_emb[h] * rel_vec * ent_emb).sum(-1)              # score EVERY entity as the tail
        top5 = scores.topk(5).indices.tolist()

    print(f"\nquery:  {id2ent[h]} --[{id2rel[r]}]--> ?    (true answer: {id2ent[t]})")
    print("encoder's top-5 predicted tails:")
    for rank, e in enumerate(top5, 1):
        print(f"  {rank}. {id2ent[e]}" + ("   <-- TRUE" if e == t else ""))
    true_rank = int((scores > scores[t]).sum()) + 1
    print(f"  (the true tail is ranked #{true_rank} of {kg['n_ent']:,})")

    print("\nreal triple vs random corruption (does real score higher?):")
    wins = 0
    with torch.no_grad():
        for _ in range(5):
            t_fake = random.randint(0, kg["n_ent"] - 1)
            s_real = (ent_emb[h] * rel_vec * ent_emb[t]).sum().item()
            s_fake = (ent_emb[h] * rel_vec * ent_emb[t_fake]).sum().item()
            wins += s_real > s_fake
            print(f"  real={s_real:+.3f}   fake={s_fake:+.3f}   "
                  f"{'real wins' if s_real > s_fake else 'FAKE wins'}")

    hint("the true tail should sit at/near the top AND the other guesses should be "
         f"the SAME TYPE of thing. Real beat a random fake {wins}/5 times = 'good at prediction'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
