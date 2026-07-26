"""The neighbourhood discriminator D_match (paper: Adversarial Generator Training).

D_match answers the second question about a candidate: "does this filler FIT
the anchor entity's neighbourhood?" It cross-attends the candidate's embedding
against a sample of the anchor's ACTUAL neighbour embeddings — deliberately
never a single pooled vector, because a d-dimensional pooled code cannot
represent membership over neighbour sets larger than d (Wagstaff et al.).

Training pairs are built purely from data:

    (anchor, its true slot-filler)                  -> label 1  (fits)
    (anchor, another anchor's same-relation filler) -> label 0  (does not fit)

Popular hub entities appear equally in both classes, so global popularity
carries no label signal — the only way to score well is to genuinely compare
the candidate against the neighbour set. The generator is trained to push this
score DOWN (produce fillers the anchor's neighbourhood does NOT corroborate),
while the plausibility discriminator keeps those fillers realistic.

The direct anchor–candidate edge is excluded from the neighbour sample by the
training-data builder; otherwise "fits" could be read off trivially.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeighbourhoodDiscriminator(nn.Module):
    """Scores whether a candidate filler belongs in an anchor's neighbourhood.

    NOTE: do not rename the attributes `q_proj`, `k_proj`, `v_proj`, `score` —
    they are the state-dict keys stored inside every saved checkpoint (the
    locked generator_*.pt artifacts carry this discriminator's weights under
    the payload key "dmatch_state").
    """

    def __init__(self, dim: int = 64, d_model: int = 64, hidden: int = 128,
                 n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(dim, d_model)   # candidate -> attention query
        self.k_proj = nn.Linear(dim, d_model)   # neighbours -> attention keys
        self.v_proj = nn.Linear(dim, d_model)   # neighbours -> attention values
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.score = nn.Sequential(
            nn.Linear(3 * d_model + 1, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, candidate_embedding: torch.Tensor,
                neighbour_embeddings: torch.Tensor,
                neighbour_mask: torch.Tensor) -> torch.Tensor:
        """Return one neighbourhood-fit logit per row, shape [batch]. High = fits.

        candidate_embedding  : [batch, dim]       frozen E' rows of candidates.
        neighbour_embeddings : [batch, N, dim]    frozen E' rows of a sample of
                                                  the anchor's neighbours.
        neighbour_mask       : [batch, N] bool    True where a real neighbour
                                                  is present (rows are padded).
        """
        batch_size, num_neighbours, _ = neighbour_embeddings.shape

        # Multi-head cross-attention: the candidate (query) looks across the
        # anchor's neighbours (keys/values) and pools what it finds relevant.
        query = self.q_proj(candidate_embedding)                    # [B, D]
        keys = self.k_proj(neighbour_embeddings)                    # [B, N, D]
        values = self.v_proj(neighbour_embeddings)                  # [B, N, D]
        query_heads = query.view(batch_size, self.n_heads, self.d_head)
        key_heads = keys.view(batch_size, num_neighbours, self.n_heads, self.d_head)
        attention = torch.einsum("bhd,bnhd->bhn",
                                 query_heads, key_heads) / (self.d_head ** 0.5)
        attention = attention.masked_fill(~neighbour_mask.unsqueeze(1),
                                          float("-inf"))
        weights = torch.softmax(attention, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)   # rows with zero neighbours
        value_heads = values.view(batch_size, num_neighbours,
                                  self.n_heads, self.d_head)
        attended = torch.einsum("bhn,bnhd->bhd",
                                weights, value_heads).reshape(batch_size, -1)

        # Max cosine similarity between the candidate and any neighbour: a
        # strong hand-built membership signal, given to the scorer explicitly.
        query_unit = torch.nn.functional.normalize(query, dim=1)
        key_unit = torch.nn.functional.normalize(keys, dim=2)
        cosine = torch.einsum("bd,bnd->bn",
                              query_unit, key_unit).masked_fill(~neighbour_mask, -1.0)
        max_cosine = cosine.max(dim=1).values.unsqueeze(1)

        features = torch.cat([query, attended, query * attended, max_cosine],
                             dim=1)
        return self.score(features).squeeze(-1)
