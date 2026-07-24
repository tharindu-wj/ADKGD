"""The RGCN context encoder (paper: Neighbourhood Context Encoding).

Runs relation-aware message passing over the whole knowledge graph and
returns ONE context vector per entity: the context table E'. E'[e] summarises
entity e's 2-hop neighbourhood, so every model that conditions on it (the
generator and both discriminators) is neighbourhood-aware by construction.

Where this sits in the pipeline: the encoder runs ONCE, during the DistMult
warm-up at the start of training (see train.py). The finished table E' is
frozen and stored in the checkpoint; corruption generation replays the cached
table and never needs the encoder — or torch_geometric — again.

Why RGCNConv and not FastRGCNConv: both are relation-aware, but FastRGCNConv
materialises a per-edge [E, dim, dim] weight tensor — on FB15K-237 that is
~8.9 GB per layer and runs out of memory. RGCNConv loops over relations
instead, which is slower but fits. Basis decomposition (num_bases) keeps the
parameter count small.
"""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import RGCNConv
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "NeighbourhoodContextEncoder requires torch_geometric (PyG):\n"
        "    pip install torch_geometric\n"
        "Only training needs PyG — corruption generation replays the cached E'."
    ) from exc


class NeighbourhoodContextEncoder(nn.Module):
    """Multi-relational GNN encoder: KG structure -> context table E'.

    Args:
        n_ent      : number of entities (rows of E').
        n_rel      : number of relations in the BASE edge list (before
                     inverse edges are added).
        dim        : embedding width of both the inputs and E'.
        num_bases  : basis-decomposition rank; capped at the effective
                     relation count. ~30 is the classic FB15K-237 setting.
        num_layers : RGCN layers = hops of context. 2 is typical.
        add_inverse: if True, append inverse edges tail -> head so context
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

        # Layer-0 input features: one learned vector per entity. Message
        # passing refines these into neighbourhood-aware context vectors.
        self.entity_embeddings = nn.Embedding(n_ent, dim)
        nn.init.normal_(self.entity_embeddings.weight, std=0.1)

        # With inverse edges the encoder sees 2*n_rel relation types:
        #   forward relation r -> edge type r
        #   inverse relation r -> edge type r + n_rel
        effective_relations = n_rel * 2 if add_inverse else n_rel
        self.eff_rel = effective_relations

        bases = min(num_bases, effective_relations)
        self.rgcn_layers = nn.ModuleList(
            RGCNConv(dim, dim, effective_relations, num_bases=bases, aggr="mean")
            for _ in range(num_layers)
        )

    def _augment_with_inverse(self, edge_index, edge_type):
        """Append inverse edges (tail -> head, relation id r + n_rel)."""
        source, target = edge_index[0], edge_index[1]
        inverse_index = torch.stack([target, source], dim=0)
        inverse_type = edge_type + self.n_rel
        full_index = torch.cat([edge_index, inverse_index], dim=1)
        full_type = torch.cat([edge_type, inverse_type], dim=0)
        return full_index, full_type

    def forward(self, edge_index, edge_type):
        """Return the context table E', shape [n_ent, dim].

        edge_index : LongTensor [2, E], base directed edges head -> tail.
        edge_type  : LongTensor [E], relation id in [0, n_rel).

        Inverse edges (if enabled) are added here, so pass the BASE edge list
        exactly as loaders.build_edge_index() produces it.
        """
        if self.add_inverse:
            edge_index, edge_type = self._augment_with_inverse(edge_index,
                                                               edge_type)

        x = self.entity_embeddings.weight
        for layer_index, layer in enumerate(self.rgcn_layers):
            x = layer(x, edge_index, edge_type)
            if layer_index < self.num_layers - 1:
                x = torch.relu(x)
        return x

    @torch.no_grad()
    def cache_embeddings(self, edge_index, edge_type):
        """Compute E' once for checkpointing — detached, on CPU.

        Called at the end of training: the resulting tensor goes into the
        checkpoint so corruption generation can look up E' rows without ever
        running PyG again.
        """
        self.eval()
        embeddings = self.forward(edge_index, edge_type)
        return embeddings.detach().cpu()

    @staticmethod
    def to_tensors(edge_index, edge_type, device=None):
        """Turn loaders.build_edge_index() lists into Long tensors on `device`.

        edge_index : [[src...], [dst...]] -> LongTensor [2, E]
        edge_type  : [rel...]             -> LongTensor [E]
        """
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long,
                                            device=device)
        edge_type_tensor = torch.as_tensor(edge_type, dtype=torch.long,
                                           device=device)
        return edge_index_tensor, edge_type_tensor
