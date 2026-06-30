"""RUNG 5 - The supervision: 'real' vs 'contradiction'.   (local, no PyG)

Question: what does the GAN learn FROM? Two opposite signals:
  * Discriminator POSITIVES = real 2-cycles (things that DO co-occur)
  * Generator RECON TARGETS = anti-symmetric templates (the contradiction r')
"""
from _common import banner, hint, load_dataset, is_dummy


def main():
    banner("RUNG 5 - The supervision: 'real' vs 'contradiction'")
    kg, data_dir, enc, gan = load_dataset()
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]

    from kgsage.gan.train import build_real_2cycle_positives
    from kgsage.gan.partner_templates import mine_partner_templates

    # the GAN trainer auto-relaxes min_support for dummy_kg; mirror that here
    min_support = 1 if is_dummy(data_dir) else 100

    pos = build_real_2cycle_positives(kg)
    print(f"\nDiscriminator POSITIVES = real 2-cycles  ({len(pos):,} found)")
    if pos:
        h, r, t, rp = pos[0]
        print(f"  anchor : {id2ent[h]} --[{id2rel[r]}]--> {id2ent[t]}")
        print(f"  partner: {id2ent[t]} --[{id2rel[rp]}]--> {id2ent[h]}")
        print("  -> this pair REALLY co-occurs (symmetric/inverse) = a 'real' example for D")

    tmpl = mine_partner_templates(kg, min_support=min_support)
    print(f"\nGenerator RECON TARGETS = anti-symmetric templates  ({len(tmpl)} relations)")
    for r in list(tmpl)[:6]:
        parts = ", ".join(f"{id2rel[rp]}({c:.2f})" for rp, c in tmpl[r][:2])
        self_swap = "   [self-swap r'=r]" if tmpl[r][0][0] == r else ""
        print(f"  {id2rel[r][:42]:42s} -> {parts}{self_swap}")

    hint("D's positives are pairs that DO co-occur; G's targets are anti-symmetric "
         "(mostly r'=r). Opposite aims is WHY the discriminator can't collapse to 2*ln2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
