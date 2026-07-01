"""RGCN context encoder for the KGSAGE conditional GAN (B1a).

WHAT IT DOES
    Runs relation-aware message passing over the whole KG and returns ONE
    context vector per entity — E'[e] summarises entity e's neighbourhood.
    The generator conditions on E'[h] and E'[t] so its corruptions are aware of
    the entity's surroundings (the "converging-context" negative: a tail that is
    type-valid but contradicts what the head's neighbourhood implies).

WHERE IT LIVES IN THE PIPELINE
    PHASE 1 only. The encoder is trained JOINTLY with the generator/discriminator
    on the graph, then the FINAL E' is cached in the checkpoint. Inference
    (PHASE 2, inside ADKGD) replays that cached E' and never imports PyG.

WHY RGCNConv (not FastRGCNConv)
    Both are relation-aware and share the same constructor. FastRGCNConv is
    faster because it processes all relations at once — but it materialises a
    per-edge [E, dim, dim] weight tensor, which on FB15K-237 (~544k directed
    edges incl. inverses, dim=64) needs ~8.9 GB PER LAYER and OOMs. RGCNConv
    instead LOOPS over relations, keeping memory at O(E * dim) — it fits
    comfortably. Basis decomposition (num_bases) keeps the parameter count small.
    The relation loop is a little slower, but correctness/fit beats speed here.
"""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import RGCNConv
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "KGSAGEEncoder requires torch_geometric (PyG). Install it with:\n"
        "    pip install torch_geometric\n"
        "Only PHASE 1 (GAN training) needs PyG — inference replays the cached E'."
    ) from exc


class KGSAGEEncoder(nn.Module):
    """Multi-relational GNN encoder: KG structure -> context embeddings E'.

    Args:
        n_ent      : number of entities (rows of E').
        n_rel      : number of relations in the BASE edge list (before inverse).
        dim        : embedding width (both input features and E').
        num_bases  : basis-decomposition rank; capped at the effective relation
                     count. ~30 is the classic FB15K-237 setting.
        num_layers : number of FastRGCNConv layers (hops of context). 2 is typical.
        add_inverse: if True, append inverse edges t -> r+n_rel -> h so context
                     flows both ways; the encoder then sees 2*n_rel relations.
    """

    def __init__(self, n_ent, n_rel, dim=64, num_bases=30, num_layers=2,
                 add_inverse=True):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.num_bases = num_bases
        self.num_layers = num_layers
        self.add_inverse = add_inverse

        # Layer-0 input features: a learned vector per entity. Message passing
        # refines these into neighbourhood-aware context vectors.
        self.node_emb = nn.Embedding(n_ent, dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)

        # With inverse edges the encoder sees 2*n_rel relation types:
        #   forward relation r  ->  edge_type r
        #   inverse relation r  ->  edge_type r + n_rel
        eff_rel = n_rel * 2 if add_inverse else n_rel
        self.eff_rel = eff_rel

        nb = min(num_bases, eff_rel)  # num_bases must not exceed relation count
        self.convs = nn.ModuleList(
            RGCNConv(dim, dim, eff_rel, num_bases=nb, aggr="mean")
            for _ in range(num_layers)
        )

    def _augment_with_inverse(self, edge_index, edge_type):
        """Append inverse edges (t -> h, relation id r + n_rel)."""
        src, dst = edge_index[0], edge_index[1]
        inv_index = torch.stack([dst, src], dim=0)
        inv_type = edge_type + self.n_rel
        full_index = torch.cat([edge_index, inv_index], dim=1)
        full_type = torch.cat([edge_type, inv_type], dim=0)
        return full_index, full_type

    def forward(self, edge_index, edge_type):
        """Return context embeddings E' of shape [n_ent, dim].

        Args:
            edge_index : LongTensor [2, E], base directed edges h -> t.
            edge_type  : LongTensor [E], relation id in [0, n_rel).

        Inverse edges (if enabled) are added here, so pass the BASE edge list
        exactly as loaders.build_edge_index() produces it.
        """
        if self.add_inverse:
            edge_index, edge_type = self._augment_with_inverse(edge_index, edge_type)

        x = self.node_emb.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < self.num_layers - 1:
                x = torch.relu(x)
        return x

    @torch.no_grad()
    def cache_embeddings(self, edge_index, edge_type):
        """Compute E' once for checkpointing — detached, on CPU.

        Used at the end of PHASE 1 training: the resulting tensor is stored in
        the checkpoint so PHASE 2 (inference) can look up E'[h], E'[t] without
        ever running PyG again.
        """
        self.eval()
        emb = self.forward(edge_index, edge_type)
        return emb.detach().cpu()

    @staticmethod
    def to_tensors(edge_index, edge_type, device=None):
        """Turn loaders.build_edge_index() lists into Long tensors on `device`.

        edge_index : [[src...], [dst...]] -> LongTensor [2, E]
        edge_type  : [rel...]             -> LongTensor [E]
        """
        ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        et = torch.as_tensor(edge_type, dtype=torch.long, device=device)
        return ei, et
