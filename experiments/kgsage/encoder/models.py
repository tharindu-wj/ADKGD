"""KGSAGE Encoder — RGCN + DistMult.

Two networks trained jointly on FB15K-237 link prediction:

  KGSAGEEncoder
    An RGCN (Relational Graph Convolutional Network).
    Reads the whole KG as a typed graph and produces a 200-dim vector for
    every entity. Each layer aggregates messages from neighbors via a
    relation-specific weight matrix, so entities that appear in similar
    relational contexts end up with similar embeddings.

  KGSAGEDistMultDecoder
    A bilinear scoring head: score(h, r, t) = <h_emb, r_emb, t_emb>.
    The decoder has its own relation embedding table (nn.Embedding) which
    gets trained via the link-prediction loss. After training, this table
    is exactly the relation embeddings KGSAGE's Phase 2 Generator needs.

WHY THIS COMBINATION:
  RGCN gives us entity embeddings that capture multi-hop graph structure.
  DistMult gives us relation embeddings as a side effect of normal KGE
  training. Together they cover what Phase 2 needs (anchor (h, r, t) →
  partner distribution over r') without forcing us to port CompGCN from
  PyTorch 1.0.

  This is exactly the pipeline Schlichtkrull et al. 2018 used in the
  original R-GCN paper for link prediction. We're not innovating on the
  encoder; we're using the standard pretraining recipe.

WHY 200 DIM:
  RGCN paper used 100. CompGCN paper used 200. DistMult paper used 200.
  200 is a fine default for FB15K-237 with 237 relations. Larger would
  be wasteful given the data size; smaller risks underfitting.

WHY 2 LAYERS:
  Layer 1 aggregates direct neighbors (1-hop).
  Layer 2 aggregates 2-hop information — which is what we ACTUALLY want
  to encode anti-symmetric structure. The role-swap contradiction is a
  2-hop pattern: h --r--> t --r'--> h. With 1 layer we'd miss this.
  Deeper networks oversmooth on FB15K-237 (well-known result).

WHY 30 BASES:
  RGCN parameterizes each W_r as a linear combination of `num_bases`
  basis matrices. With 237 relations and 200 dim, full-rank would mean
  237 * 200 * 200 = 9.5M parameters per layer. With 30 bases we get
  30 * 200 * 200 + 237 * 30 = 1.2M parameters per layer — 8x smaller,
  same expressivity in practice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv


class KGSAGEEncoder(nn.Module):
    """RGCN backbone — produces entity embeddings from the typed KG.

    Inputs at every forward pass:
      edge_index  : (2, num_edges) — the KG's edges
      edge_type   : (num_edges,)   — relation ID for each edge

    These come from the loaded KG and stay fixed throughout training. The
    only thing that changes is the embedding lookup at the start and the
    RGCN layer weights. Output is (n_ent, dim) — one row per entity.
    """

    def __init__(self, n_ent, n_rel, dim=200, n_layers=2, num_bases=30):
        super().__init__()

        # Initial entity embeddings — these are the input to layer 1.
        # RGCN updates them by message passing; the trained values are
        # what we save for Phase 2.
        self.ent_emb = nn.Embedding(n_ent, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)

        # RGCN layers. Each layer:
        #   - Reads (x, edge_index, edge_type)
        #   - For each edge (u, v) with relation r, sends W_r * x[u] -> x[v]
        #   - Aggregates by averaging incoming messages (per relation)
        # We add self-loops via the layer's `aggr` default; PyG handles it.
        self.layers = nn.ModuleList([
            RGCNConv(dim, dim, num_relations=n_rel, num_bases=num_bases)
            for _ in range(n_layers)
        ])

        # Save sizes so checkpoint loader can rebuild the model.
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.n_layers = n_layers
        self.num_bases = num_bases

    def forward(self, edge_index, edge_type):
        """Run the encoder.

        Returns (n_ent, dim) — entity embeddings updated by message passing.
        """
        # Layer 0: just the embedding lookup.
        x = self.ent_emb.weight

        # Layers 1..n_layers: relation-specific message passing + ReLU.
        # Last layer skips the ReLU — typical pattern, leaves output free
        # to be positive or negative, which DistMult expects.
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        return x


class KGSAGEDistMultDecoder(nn.Module):
    """DistMult scoring head + the relation embedding table.

    score(h, r, t) = sum(h_emb * r_emb * t_emb)   (element-wise product, then sum)

    DistMult is symmetric in (h, t): swapping head and tail gives the same
    score. This is a known weakness for asymmetric relations — but it
    DOES NOT hurt our use case, because Phase 2 doesn't use the decoder
    at all. We only need the relation embedding TABLE, which DistMult
    training produces.

    The relation embeddings end up encoding "what shape of head/tail
    distribution this relation prefers", which is exactly what Phase 2's
    KGSAGE Generator needs to predict contradicting partner relations.
    """

    def __init__(self, n_rel, dim=200):
        super().__init__()

        # The reusable artifact — Phase 2 loads this directly.
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.n_rel = n_rel
        self.dim = dim

    def score(self, h_emb, r_id, t_emb):
        """Score a batch of triples given their embeddings.

        Inputs:
          h_emb : (batch, dim) — head entity embeddings (from encoder output)
          r_id  : (batch,)     — relation IDs
          t_emb : (batch, dim) — tail entity embeddings

        Returns: (batch,) scores. Higher = more plausible.
        """
        r_emb = self.rel_emb(r_id)
        return (h_emb * r_emb * t_emb).sum(dim=-1)


class KGSAGELinkPredictor(nn.Module):
    """Combined encoder + decoder for end-to-end training.

    Held together purely for training convenience. After training, we
    save the two sub-modules' weights separately — Phase 2 loads them
    via `load_pretrained()`.
    """

    def __init__(self, n_ent, n_rel, dim=200, n_layers=2, num_bases=30):
        super().__init__()
        self.encoder = KGSAGEEncoder(
            n_ent, n_rel, dim=dim, n_layers=n_layers, num_bases=num_bases,
        )
        self.decoder = KGSAGEDistMultDecoder(n_rel, dim=dim)

    def forward(self, edge_index, edge_type, triples):
        """Score a batch of triples.

        Inputs:
          edge_index, edge_type : the whole KG (used for encoding)
          triples               : (batch, 3) tensor of (h, r, t) IDs to score

        Returns: (batch,) scores.
        """
        # Encode the entire KG once per forward pass.
        entity_embeddings = self.encoder(edge_index, edge_type)

        # Look up the specific entities we need for this batch.
        h_emb = entity_embeddings[triples[:, 0]]
        t_emb = entity_embeddings[triples[:, 2]]
        r_id = triples[:, 1]

        return self.decoder.score(h_emb, r_id, t_emb)

    def save_pretrained(self, path, ent2id, rel2id):
        """Save everything Phase 2 needs to recover the embeddings.

        Saves a checkpoint with:
          - encoder state_dict (so we can re-run the encoder if needed)
          - decoder state_dict (contains the relation embeddings)
          - cached final entity embeddings (so Phase 2 doesn't need to
            re-run message passing every time)
          - cached final relation embeddings
          - vocab maps (ent2id, rel2id) so Phase 2 can translate strings
        """
        # NOTE: caller must provide the trained model already in eval mode
        # and on whichever device they want to save from.
        payload = {
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "ent2id": ent2id,
            "rel2id": rel2id,
            "config": {
                "n_ent": self.encoder.n_ent,
                "n_rel": self.encoder.n_rel,
                "dim": self.encoder.dim,
                "n_layers": self.encoder.n_layers,
                "num_bases": self.encoder.num_bases,
            },
        }
        torch.save(payload, path)

    @staticmethod
    def load_pretrained(path, device=None):
        """Reconstruct the link predictor from a saved checkpoint.

        Returns:
            model     : KGSAGELinkPredictor with weights loaded, in eval mode
            ent2id    : dict
            rel2id    : dict
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]

        model = KGSAGELinkPredictor(
            n_ent=config["n_ent"],
            n_rel=config["n_rel"],
            dim=config["dim"],
            n_layers=config["n_layers"],
            num_bases=config["num_bases"],
        ).to(device)
        model.encoder.load_state_dict(payload["encoder_state"])
        model.decoder.load_state_dict(payload["decoder_state"])
        model.eval()

        return model, payload["ent2id"], payload["rel2id"]
