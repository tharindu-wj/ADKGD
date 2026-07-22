"""D_match: the neighbourhood-consistency discriminator of KGSAGE-2.

Answers "does candidate x belong to anchor a's world?" by CROSS-ATTENDING the
candidate embedding against a sample of the anchor's ACTUAL neighbour
embeddings -- never a pooled code (Wagstaff bound). Trained GAN-CLS style on
pairs built purely from data:

    (anchor, its true slot-filler)                  -> 1   (fits)
    (anchor, another anchor's same-relation filler) -> 0   (mismatched)

Hubs occur in both classes symmetrically, so global popularity carries no
label signal by construction -- the only winning strategy is genuine
set-comparison. The generator will be trained to DRIVE THIS SCORE DOWN
(produce candidates alien to the anchor's world) while the realism discriminator
keeps them plausible.

The direct edge anchor--candidate is EXCLUDED from the neighbour sample by the
data builder (otherwise membership is trivial string-matching); the discriminator
must detect co-neighbourhood structure, which is exactly the corroboration
signal (untrained max-cosine proxy on E' already reaches AUC 0.80).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DMatch(nn.Module):
    def __init__(self, dim: int = 64, d_model: int = 64, hidden: int = 128,
                 n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(dim, d_model)
        self.k_proj = nn.Linear(dim, d_model)
        self.v_proj = nn.Linear(dim, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.score = nn.Sequential(
            nn.Linear(3 * d_model + 1, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand_emb: torch.Tensor, nbr_emb: torch.Tensor,
                nbr_mask: torch.Tensor) -> torch.Tensor:
        """[B] logits: high = candidate fits this neighbour set.

        cand_emb : [B, dim]      frozen E' rows of the candidates
        nbr_emb  : [B, N, dim]   frozen E' rows of sampled anchor neighbours
        nbr_mask : [B, N] bool   True where a real neighbour is present
        """
        B, N, _ = nbr_emb.shape
        q = self.q_proj(cand_emb)                                  # [B, D]
        k = self.k_proj(nbr_emb)                                   # [B, N, D]
        v = self.v_proj(nbr_emb)
        qh = q.view(B, self.n_heads, self.d_head)
        kh = k.view(B, N, self.n_heads, self.d_head)
        att = torch.einsum("bhd,bnhd->bhn", qh, kh) / (self.d_head ** 0.5)
        att = att.masked_fill(~nbr_mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(att, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)                # rows with zero neighbours
        vh = v.view(B, N, self.n_heads, self.d_head)
        pooled = torch.einsum("bhn,bnhd->bhd", w, vh).reshape(B, -1)

        # max cosine similarity: the strong untrained baseline, given explicitly
        qc = torch.nn.functional.normalize(q, dim=1)
        kc = torch.nn.functional.normalize(k, dim=2)
        sim = torch.einsum("bd,bnd->bn", qc, kc).masked_fill(~nbr_mask, -1.0)
        maxsim = sim.max(dim=1).values.unsqueeze(1)

        feats = torch.cat([q, pooled, q * pooled, maxsim], dim=1)
        return self.score(feats).squeeze(-1)
