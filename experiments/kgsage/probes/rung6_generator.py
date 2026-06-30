"""RUNG 6 - Watch the generator choose a partner.   (local, no PyG)

Question: given an anchor, what contradiction does the trained GAN propose, and
does the symmetric-safety check work? Uses only the GAN checkpoint (its own
vocab + real-triple set), so it never needs the encoder or PyG.
"""
from _common import banner, hint, resolve_bundle


def main():
    banner("RUNG 6 - Watch the generator choose a partner")
    data_dir, enc, gan = resolve_bundle()
    print(f"[probe] gan ckpt: {gan or '(none)'}")
    if gan is None:
        print("\n[skip] No GAN checkpoint. Train one:")
        print("       python -m kgsage.cli.train_gan --data data/dummy_kg --epochs 5 "
              "--device cpu --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt")
        return 0

    import torch
    from kgsage.inference import load_kgsage_checkpoint

    p = load_kgsage_checkpoint(gan, device="cpu")
    G, id2ent, id2rel, real = p["generator"], p["id2ent"], p["id2rel"], p["real_triple_set"]
    reals = list(real)

    # an ANTI-SYMMETRIC anchor (its reverse is NOT in the graph)
    anchor = next(((h, r, t) for (h, r, t) in reals
                   if h != t and (t, r, h) not in real), reals[0])
    h, r, t = anchor
    logits = G(torch.tensor([h]), torch.tensor([r]), torch.tensor([t]))[0]
    top5 = logits.topk(min(5, logits.size(0))).indices.tolist()

    print(f"\nANCHOR (anti-symmetric):  {id2ent[h]} --[{id2rel[r]}]--> {id2ent[t]}")
    print("generator's top-5 partner relations r':")
    for e in top5:
        print(f"  {id2rel[e]}")
    rp = int(logits.argmax())
    cand = (t, rp, h)
    print(f"\n-> candidate contradiction:  {id2ent[t]} --[{id2rel[rp]}]--> {id2ent[h]}")
    print(f"   in real graph? {cand in real}    (False => it IS a contradiction)")

    # a SYMMETRIC anchor (its reverse IS in the graph) - watch the rejection logic
    sym = next(((h, r, t) for (h, r, t) in reals if h != t and (t, r, h) in real), None)
    if sym:
        h, r, t = sym
        print(f"\nNow a SYMMETRIC anchor:  {id2ent[h]} --[{id2rel[r]}]--> {id2ent[t]}")
        print(f"   reverse {id2ent[t]} --[{id2rel[r]}]--> {id2ent[h]} in graph? "
              f"{(t, r, h) in real}")
        print("   -> the self-reverse would be REJECTED at inference (it's a real fact, not a contradiction)")

    hint("the generator concentrates probability on a few partner relations; its "
         "(t,r',h) is ABSENT from the graph. For symmetric anchors the reverse IS "
         "present, so it gets correctly rejected -- that's the symmetric-safety mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
