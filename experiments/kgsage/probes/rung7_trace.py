"""RUNG 7 - Trace ONE triple end-to-end.

Takes a single real triple and walks it through every rung: anti-symmetry ->
supervision -> generated contradiction (-> encoder rank, if the encoder is
available). The local rungs always run; the encoder rung self-skips off-HPC.
"""
import os
from collections import defaultdict
from _common import banner, hint, load_dataset, is_dummy


def main():
    banner("RUNG 7 - Trace one triple end-to-end")
    kg, data_dir, enc, gan = load_dataset()
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]

    # pick one non-self-loop fact
    h, r, t = next((x for x in kg["triples_train"] if x[0] != x[2]), kg["triples_train"][0])
    print(f"\nTRIPLE:  {id2ent[h]} --[{id2rel[r]}]--> {id2ent[t]}\n")

    # [Rung 2] anti-symmetric?
    edges = defaultdict(set)
    for a, b, c in kg["triples_train"]:
        edges[(a, c)].add(b)
    anti = r not in edges.get((t, h), ())
    print(f"  [Rung 2] reverse present? {not anti}  -> "
          f"{'anti-symmetric (contradiction target)' if anti else 'symmetric (real)'}")

    # [Rung 5] has an anti-symmetric template?
    from kgsage.gan.partner_templates import mine_partner_templates
    ms = 1 if is_dummy(data_dir) else 100
    tmpl = mine_partner_templates(kg, min_support=ms)
    print(f"  [Rung 5] relation has an anti-symmetric template? {r in tmpl}")

    # [Rung 6] generator's contradiction (string-bridge into the GAN's vocab)
    if gan:
        import torch
        from kgsage.inference import load_kgsage_checkpoint
        p = load_kgsage_checkpoint(gan, device="cpu")
        try:
            gh = p["ent2id"][id2ent[h]]; gr = p["rel2id"][id2rel[r]]; gt = p["ent2id"][id2ent[t]]
            logits = p["generator"](torch.tensor([gh]), torch.tensor([gr]), torch.tensor([gt]))[0]
            rp = int(logits.argmax())
            cand = (gt, rp, gh)
            print(f"  [Rung 6] generator -> {id2ent[t]} --[{p['id2rel'][rp]}]--> {id2ent[h]}  "
                  f"(absent from graph? {cand not in p['real_triple_set']})")
        except KeyError:
            print("  [Rung 6] GAN checkpoint vocab != this dataset -- skipped "
                  "(run on the dataset the GAN was trained on)")
    else:
        print("  [Rung 6] no GAN checkpoint -- skipped")

    # [Rung 3] encoder rank of the true tail (HPC)
    if enc:
        try:
            import torch
            from kgsage.encoder.models import KGSAGELinkPredictor
            m, _, _ = KGSAGELinkPredictor.load_pretrained(enc, device="cpu")
            with torch.no_grad():
                ee = m.encoder(kg["edge_index"], kg["edge_type"])
                sc = (ee[h] * m.decoder.rel_emb.weight[r] * ee).sum(-1)
                rank = int((sc > sc[t]).sum()) + 1
            print(f"  [Rung 3] encoder rank of the true tail: #{rank} of {kg['n_ent']:,}")
        except Exception:
            print("  [Rung 3] encoder needs torch_geometric (HPC) -- skipped")
    else:
        print("  [Rung 3] no encoder checkpoint -- skipped")

    hint("you followed ONE fact: raw triple -> anti-symmetry -> supervision -> "
         "generated contradiction (-> encoder rank on HPC). That is the whole pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
