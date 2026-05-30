"""Conditional GAN for KG triple corruption (extracted from the notebook).

Source: `src/knowledge_graph_clone.ipynb` cell 14. Extracting here so the
training and sampling scripts can `import` instead of executing notebook
cells. Architecture and hyperparameter SHAPES are unchanged - the only
adjustment is the default `embedding_dim` / `bottleneck_hidden_dim` to
sensible FB15k-237 values, with overrides exposed via constructor args.

The pipeline (Pix2Pix-flavoured cGAN):
  * `EntityEmbedding` and `RelationEmbedding` - lookup tables, jointly
    trained as part of the generator's parameter list.
  * `TripleGenerator` - takes a clean triple embedding (B, 3, d) plus a
    latent noise vector (B, d_z), returns logits over the entity and
    relation vocabularies for each of the three triple positions plus
    near-one-hot Gumbel-Softmax soft samples for differentiable re-embedding.
  * `TripleDiscriminator` - cGAN critic: scores (clean, candidate) pairs.
    Spectral-norm on every Linear stabilises training.
  * `gumbel_softmax_sample` - free function, the differentiable categorical
    sampler used by G and re-used at inference.
  * `re_embed_soft_samples` - turns the three soft samples back into a
    (B, 3, d) embedding via `soft_sample @ embedding_weight`. Differentiable
    so gradients flow back through G AND through the embeddings.
"""
from __future__ import annotations

import torch
from torch import nn


_DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE = 0.2


class EntityEmbedding(nn.Module):
    """Learnable lookup table: row i is the embedding vector for entity i."""

    def __init__(self, num_entities: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.normal_(
            self.embedding_table.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5),
        )

    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(entity_indices)

    def lookup_weight(self) -> torch.Tensor:
        return self.embedding_table.weight


class RelationEmbedding(nn.Module):
    """Same shape as EntityEmbedding but for relations (much smaller vocab)."""

    def __init__(self, num_relations: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_relations, embedding_dim)
        nn.init.normal_(
            self.embedding_table.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5),
        )

    def forward(self, relation_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(relation_indices)

    def lookup_weight(self) -> torch.Tensor:
        return self.embedding_table.weight


def gumbel_softmax_sample(
    logits: torch.Tensor,
    gumbel_temperature: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Differentiable categorical sample (Jang 2017).

    At low temperatures the soft sample is almost argmax; at high
    temperatures it's almost uniform. We anneal from 1.0 to 0.5 during
    training.
    """
    uniform_noise = torch.rand_like(logits).clamp_(epsilon, 1.0 - epsilon)
    gumbel_noise  = -torch.log(-torch.log(uniform_noise))
    perturbed_logits = (logits + gumbel_noise) / gumbel_temperature
    return torch.softmax(perturbed_logits, dim=-1)


class TripleGenerator(nn.Module):
    """cGAN generator: maps (clean triple embedding, latent z) to triple logits.

    Architecture: 1x1 Conv2d encoder -> multi-head self-attention over the
    three positions -> bottleneck (concat with z) -> three independent
    MLP heads producing logits over entity / relation / entity vocabs.

    Defaults are tuned for FB15k-237 (~14k entities, 237 relations). Override
    the constructor args for other graph sizes.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        latent_dim: int = 32,
        encoder_base_channels: int = 32,
        encoder_num_conv_layers: int = 3,
        encoder_num_attn_heads: int = 4,
        bottleneck_hidden_dim: int = 512,
        leaky_relu_negative_slope: float = _DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.latent_dim    = latent_dim

        # ── Encoder: 1x1 Conv2d stack on the (B, 1, 3, d) "1-channel image" ──
        conv_layers = []
        current_in_channels  = 1
        current_out_channels = encoder_base_channels
        for _ in range(encoder_num_conv_layers):
            conv_layers.append(
                nn.Conv2d(current_in_channels, current_out_channels, kernel_size=(1, 1))
            )
            conv_layers.append(nn.LeakyReLU(leaky_relu_negative_slope, inplace=True))
            current_in_channels  = current_out_channels
            current_out_channels = current_out_channels * 2
        self.encoder_conv_stack = nn.Sequential(*conv_layers)
        encoder_final_channel_count = current_in_channels

        # After the conv stack: (B, C_final, 3, d). Flatten per token, project to bottleneck.
        self.per_token_projection = nn.Linear(
            encoder_final_channel_count * embedding_dim, bottleneck_hidden_dim,
        )
        self.self_attention_layer = nn.MultiheadAttention(
            embed_dim=bottleneck_hidden_dim,
            num_heads=encoder_num_attn_heads,
            batch_first=True,
        )
        self.attention_layer_norm = nn.LayerNorm(bottleneck_hidden_dim)

        # ── Bottleneck: pool the 3 token features, concat with latent z ──
        self.bottleneck_projection = nn.Linear(
            bottleneck_hidden_dim * 3 + latent_dim,
            bottleneck_hidden_dim,
        )

        # ── Three output heads: one per triple position ──
        self.head_entity_predictor = nn.Sequential(
            nn.Linear(bottleneck_hidden_dim, bottleneck_hidden_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden_dim, num_entities),
        )
        self.relation_predictor = nn.Sequential(
            nn.Linear(bottleneck_hidden_dim, bottleneck_hidden_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden_dim, num_relations),
        )
        self.tail_entity_predictor = nn.Sequential(
            nn.Linear(bottleneck_hidden_dim, bottleneck_hidden_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_hidden_dim, num_entities),
        )

    def forward(
        self,
        clean_triple_embedding: torch.Tensor,
        latent_sample: torch.Tensor,
        gumbel_temperature: float = 1.0,
        return_soft_samples: bool = True,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        clean_triple_embedding : (B, 3, d) float - h, r, t embeddings stacked
        latent_sample          : (B, d_z) float - per-sample noise vector
        gumbel_temperature     : softmax temperature for the Gumbel sampler
        return_soft_samples    : if True, also return near-one-hot Gumbel samples
        """
        batch_size = clean_triple_embedding.shape[0]

        # (B, 1, 3, d) -> conv stack -> (B, C_final, 3, d)
        encoder_input = clean_triple_embedding.unsqueeze(1)
        encoder_conv_output = self.encoder_conv_stack(encoder_input)
        # Treat each of the 3 positions as a token: (B, 3, C_final*d)
        per_token_features = (
            encoder_conv_output
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, 3, -1)
        )
        projected_tokens = self.per_token_projection(per_token_features)  # (B, 3, hidden)
        attention_output, _ = self.self_attention_layer(
            projected_tokens, projected_tokens, projected_tokens,
        )
        refined_tokens = self.attention_layer_norm(projected_tokens + attention_output)

        flattened_token_features = refined_tokens.flatten(1)               # (B, 3*hidden)
        bottleneck_input = torch.cat(
            [flattened_token_features, latent_sample], dim=1,
        )                                                                   # (B, 3*hidden + d_z)
        bottleneck_hidden_state = self.bottleneck_projection(bottleneck_input)

        head_entity_logits = self.head_entity_predictor(bottleneck_hidden_state)
        relation_logits    = self.relation_predictor(bottleneck_hidden_state)
        tail_entity_logits = self.tail_entity_predictor(bottleneck_hidden_state)

        generator_output = {
            'head_entity_logits': head_entity_logits,
            'relation_logits':    relation_logits,
            'tail_entity_logits': tail_entity_logits,
        }
        if return_soft_samples:
            generator_output['head_entity_soft_sample'] = gumbel_softmax_sample(
                head_entity_logits, gumbel_temperature,
            )
            generator_output['relation_soft_sample'] = gumbel_softmax_sample(
                relation_logits, gumbel_temperature,
            )
            generator_output['tail_entity_soft_sample'] = gumbel_softmax_sample(
                tail_entity_logits, gumbel_temperature,
            )
        return generator_output


class TripleDiscriminator(nn.Module):
    """cGAN critic: scores (clean, candidate) embedding pairs.

    Spectral-norm on every Linear caps each layer's Lipschitz constant,
    preventing D from "winning" too fast.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        leaky_relu_negative_slope: float = _DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE,
    ) -> None:
        super().__init__()
        # Input: two (B, 3, d) embeddings flattened and concatenated => (B, 6*d).
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(6 * embedding_dim, hidden_dim)),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace=True),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(leaky_relu_negative_slope, inplace=True),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )

    def forward(
        self,
        clean_triple_embedding: torch.Tensor,
        candidate_triple_embedding: torch.Tensor,
    ) -> torch.Tensor:
        discriminator_input = torch.cat(
            [clean_triple_embedding.flatten(1), candidate_triple_embedding.flatten(1)],
            dim=1,
        )
        return self.mlp(discriminator_input)


def re_embed_soft_samples(
    head_entity_soft_sample: torch.Tensor,
    relation_soft_sample: torch.Tensor,
    tail_entity_soft_sample: torch.Tensor,
    entity_weight: torch.Tensor,
    relation_weight: torch.Tensor,
) -> torch.Tensor:
    """Differentiable round-trip from Gumbel soft samples back to (B, 3, d).

    `soft_sample @ entity_weight ~= entity_weight[argmax(soft_sample)]` but
    differentiable, so gradients flow back through both G and the embeddings.
    """
    head_entity_embedding = head_entity_soft_sample @ entity_weight
    relation_embedding    = relation_soft_sample    @ relation_weight
    tail_entity_embedding = tail_entity_soft_sample @ entity_weight
    return torch.stack(
        [head_entity_embedding, relation_embedding, tail_entity_embedding],
        dim=1,
    )


def build_triple_embedding(
    triple_indices: torch.Tensor,
    entity_embedding_module: EntityEmbedding,
    relation_embedding_module: RelationEmbedding,
) -> torch.Tensor:
    """Look up (h, r, t) embeddings for a batch of integer-indexed triples.

    Parameters
    ----------
    triple_indices : (B, 3) long - [head_idx, relation_idx, tail_idx] per row
    returns        : (B, 3, embedding_dim) float
    """
    return torch.stack(
        [
            entity_embedding_module(triple_indices[:, 0]),
            relation_embedding_module(triple_indices[:, 1]),
            entity_embedding_module(triple_indices[:, 2]),
        ],
        dim=1,
    )
