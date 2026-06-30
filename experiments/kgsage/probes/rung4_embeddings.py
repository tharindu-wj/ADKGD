"""RUNG 4 - What did the embeddings learn?

4a (relation geometry) reads ONLY the relation table from the encoder checkpoint
   -> no RGCN, no torch_geometric needed (just the .pt file).
4b (entity nearest-neighbours) runs the RGCN forward -> needs torch_geometric (HPC).
"""
from _common import banner, hint, load_dataset


def main():
    banner("RUNG 4 - What did the embeddings learn?")
    kg, data_dir, enc, gan = load_dataset()

    if enc is None:
        print("\n[skip] No encoder checkpoint; train it first (HPC) then re-run.")
        return 0

    import torch
    import torch.nn.functional as F

    # ---- 4a: relation geometry (no PyG; just load the relation table) ----
    ck = torch.load(enc, map_location="cpu", weights_only=False)
    rel_emb = ck["decoder_state"]["rel_emb.weight"]          # (n_rel, dim)
    id2rel = {i: s for s, i in ck["rel2id"].items()}
    n_rel = rel_emb.size(0)

    print("\n4a - relation geometry  (cosine similarity; NO RGCN needed):")
    unit = F.normalize(rel_emb, dim=1)
    sims = unit @ unit.t()
    eye = torch.eye(n_rel, dtype=torch.bool)
    sims_hi = sims.masked_fill(eye, -2.0)   # exclude self from 'closest'  (argmax)
    sims_lo = sims.masked_fill(eye, 2.0)    # exclude self from 'opposite' (argmin)
    for r in range(min(6, n_rel)):
        nn = int(sims_hi[r].argmax())
        op = int(sims_lo[r].argmin())
        print(f"  {id2rel[r][:44]}")
        print(f"       closest : {id2rel[nn][:44]:44s} cos {sims_hi[r][nn]:+.2f}")
        print(f"       opposite: {id2rel[op][:44]:44s} cos {sims_lo[r][op]:+.2f}")
    hint("related relations sit close (high cos); anti-symmetric pairs sit far / "
         "opposite. This is Test 1.2 -- but you're reading actual cosines, not a p-value.")

    # ---- 4b: entity nearest-neighbours (needs the RGCN forward) ----
    try:
        from kgsage.encoder.models import KGSAGELinkPredictor
        model, ent2id, _ = KGSAGELinkPredictor.load_pretrained(enc, device="cpu")
        id2ent = {i: s for s, i in ent2id.items()}
        with torch.no_grad():
            ent_emb = model.encoder(kg["edge_index"], kg["edge_type"])
        e0 = kg["triples_train"][0][0]
        ne = F.normalize(ent_emb, dim=1)
        sim = ne[e0] @ ne.t()
        sim[e0] = -2.0
        print(f"\n4b - nearest entities to '{id2ent[e0]}':")
        for e in sim.topk(5).indices.tolist():
            print(f"  {id2ent[e]:40s} cos {sim[e]:+.2f}")
        hint("the 5 nearest entities should be the SAME KIND of thing -- proof the "
             "RGCN organised entity space by meaning.")
    except Exception as e:
        print(f"\n[4b skip] {type(e).__name__}: entity neighbours need the RGCN forward "
              f"(torch_geometric, HPC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
