"""Alternating G/D training loop for the TripleGenerator + TripleDiscriminator.

Extracted from `src/knowledge_graph_clone.ipynb` cell 18. Same loss structure
(`L_total = 20*L_recon + 1*L_adv + 0.5*L_div`), same Gumbel temperature anneal,
same instance noise schedule. The notebook had these as module-level globals
in one big cell; here they're parameters of a single `train_triple_gan`
function so the sampling script can stand alone.

Training data: `(clean, target)` index tensors built by
`dataset_builders.build_triple_pair_dataset` from TRIC-corrupted versions
of the union KG. The notebook narrows TRIC to `change_relation` only (good
for the dummy KG); for FB15k-237 we keep the default mix so the type-aware
`swap_subject_same_type` / `swap_object_same_type` operations are active
- those are exactly the type-coherent-but-wrong supervision signal we want.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data as data_utils

from corruption_strategies import apply_tric_corruption
from dataset_builders import build_triple_pair_dataset
from models.triple_gan import (
    EntityEmbedding,
    RelationEmbedding,
    TripleDiscriminator,
    TripleGenerator,
    build_triple_embedding,
    re_embed_soft_samples,
)


@dataclass
class TrainConfig:
    """All knobs the training loop needs. Sensible FB15k-237 defaults."""
    # Architecture sizes (passed straight to TripleGenerator / TripleDiscriminator)
    embedding_dim:           int = 200
    latent_dim:              int = 32
    encoder_base_channels:   int = 32
    encoder_num_conv_layers: int = 3
    encoder_num_attn_heads:  int = 4
    bottleneck_hidden_dim:   int = 512
    discriminator_hidden_dim: int = 256

    # TRIC pair generation
    num_corruption_steps_per_round: int = 2
    num_training_rounds:            int = 4000
    # If None, use the default TRIC mix (which activates swap-same-type when
    # node_types is provided). Set e.g. {"change_relation": 1} to restrict
    # supervision to a single op (the notebook did this for the dummy KG).
    tric_operation_weights:         Optional[Dict[str, float]] = None
    permute_entities:               bool = True

    # Optimisation
    total_epochs:                   int = 400
    training_batch_size:             int = 64
    learning_rate:                  float = 1e-4
    discriminator_lr_multiplier:    float = 0.25
    adam_beta1:                     float = 0.5
    adam_beta2:                     float = 0.999
    gradient_clip_max_norm:         float = 5.0

    # Loss weights
    loss_weight_reconstruction:     float = 20.0
    loss_weight_adversarial:        float = 1.0
    loss_weight_diversity:          float = 0.5

    # Stabilisers
    real_label_value:               float = 0.9
    fake_label_value:               float = 0.0
    instance_noise_std_start:       float = 0.1
    instance_noise_std_end:         float = 0.01
    gumbel_temperature_start:       float = 1.0
    gumbel_temperature_end:         float = 0.5

    # Logging / checkpoint
    log_every_n_epochs:             int = 15
    checkpoint_every_n_epochs:      int = 0  # 0 = only at the end
    random_seed:                    int = 0


@dataclass
class TrainHistory:
    """Per-step (G-step) loss / score histories - same arrays the notebook plots."""
    g_adversarial:            List[float] = field(default_factory=list)
    g_reconstruction:         List[float] = field(default_factory=list)
    g_diversity:              List[float] = field(default_factory=list)
    g_total:                  List[float] = field(default_factory=list)
    d_total:                  List[float] = field(default_factory=list)
    d_score_on_real:          List[float] = field(default_factory=list)
    d_score_on_fake_before_g: List[float] = field(default_factory=list)
    d_score_on_fake_after_g:  List[float] = field(default_factory=list)
    instance_noise_std:       List[float] = field(default_factory=list)  # per-epoch
    gumbel_temperature:       List[float] = field(default_factory=list)  # per-epoch


def _add_instance_noise(
    clean_embedding: torch.Tensor,
    candidate_embedding: torch.Tensor,
    noise_std: float,
):
    if noise_std <= 0:
        return clean_embedding, candidate_embedding
    return (
        clean_embedding     + torch.randn_like(clean_embedding)     * noise_std,
        candidate_embedding + torch.randn_like(candidate_embedding) * noise_std,
    )


def _mode_seeking_diversity_loss(
    candidate_1: torch.Tensor,
    candidate_2: torch.Tensor,
    latent_1: torch.Tensor,
    latent_2: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Mao et al. CVPR 2019: penalise G if z is ignored."""
    candidate_difference_magnitude = (candidate_1 - candidate_2).abs().mean()
    latent_difference_magnitude    = (latent_1 - latent_2).abs().mean()
    return -candidate_difference_magnitude / (latent_difference_magnitude + epsilon)


def train_triple_gan(
    *,
    clean_triples: List[tuple],
    num_entities: int,
    num_relations: int,
    node_types: Optional[Dict[int, str]],
    config: TrainConfig,
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: str = "triple_gan",
) -> Dict:
    """Train the cGAN end-to-end and return everything sampling needs.

    Returns a dict with keys:
      * generator, discriminator, entity_embedding, relation_embedding - the modules
      * history - TrainHistory
      * config  - the TrainConfig that was used
    """
    random_generator = np.random.default_rng(config.random_seed)
    torch.manual_seed(config.random_seed)

    # ── Build the (clean, target) training-pair tensor via TRIC ──────────
    tric_corruption_function = functools.partial(
        apply_tric_corruption,
        num_entities=num_entities,
        num_relations=num_relations,
        node_types=node_types,
        operation_weights=config.tric_operation_weights,
    )
    triple_pair_tensor = build_triple_pair_dataset(
        clean_triples,
        num_rounds=config.num_training_rounds,
        num_corruption_steps_per_round=config.num_corruption_steps_per_round,
        random_generator=random_generator,
        num_entities=num_entities,
        num_relations=num_relations,
        corruption_fn=tric_corruption_function,
        permute_entities=config.permute_entities,
    )
    training_clean_triples  = triple_pair_tensor[:, :3]
    training_target_triples = triple_pair_tensor[:, 3:]
    triple_pair_torch_dataset = data_utils.TensorDataset(
        training_clean_triples, training_target_triples,
    )
    triple_pair_dataloader = data_utils.DataLoader(
        triple_pair_torch_dataset,
        batch_size=config.training_batch_size,
        shuffle=True, drop_last=True,
    )

    if len(triple_pair_dataloader) == 0:
        raise RuntimeError(
            f"No training pairs produced from {len(clean_triples)} clean triples. "
            f"Increase num_training_rounds or num_corruption_steps_per_round."
        )

    # ── Instantiate modules ───────────────────────────────────────────────
    entity_embedding   = EntityEmbedding(num_entities,  config.embedding_dim).to(device)
    relation_embedding = RelationEmbedding(num_relations, config.embedding_dim).to(device)
    generator_model = TripleGenerator(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config.embedding_dim,
        latent_dim=config.latent_dim,
        encoder_base_channels=config.encoder_base_channels,
        encoder_num_conv_layers=config.encoder_num_conv_layers,
        encoder_num_attn_heads=config.encoder_num_attn_heads,
        bottleneck_hidden_dim=config.bottleneck_hidden_dim,
    ).to(device)
    discriminator_model = TripleDiscriminator(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.discriminator_hidden_dim,
    ).to(device)

    binary_cross_entropy_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    cross_entropy_loss                    = nn.CrossEntropyLoss().to(device)

    generator_parameter_list = (
        list(generator_model.parameters())
        + list(entity_embedding.parameters())
        + list(relation_embedding.parameters())
    )
    generator_optimizer = optim.Adam(
        generator_parameter_list,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    discriminator_optimizer = optim.Adam(
        discriminator_model.parameters(),
        lr=config.learning_rate * config.discriminator_lr_multiplier,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    history = TrainHistory()
    print(
        f"Training: {config.total_epochs} epochs, "
        f"{len(triple_pair_dataloader)} batches/epoch, "
        f"{len(triple_pair_torch_dataset)} pairs",
        flush=True,
    )
    print(
        f"loss_weights: recon={config.loss_weight_reconstruction}, "
        f"adv={config.loss_weight_adversarial}, div={config.loss_weight_diversity}",
        flush=True,
    )
    print(
        f"G lr={config.learning_rate:.1e}, "
        f"D lr={config.learning_rate * config.discriminator_lr_multiplier:.1e}",
        flush=True,
    )
    print("-" * 70, flush=True)

    generator_model.train()
    discriminator_model.train()

    for epoch_index in range(config.total_epochs):
        epoch_d_loss_sum   = 0.0
        epoch_d_loss_count = 0

        epoch_progress_fraction = epoch_index / max(config.total_epochs - 1, 1)
        current_instance_noise_std = (
            config.instance_noise_std_start * (1 - epoch_progress_fraction)
            + config.instance_noise_std_end * epoch_progress_fraction
        )
        history.instance_noise_std.append(current_instance_noise_std)

        anneal_progress_fraction = min(1.0, epoch_index / max(1, config.total_epochs // 2))
        current_gumbel_temperature = (
            config.gumbel_temperature_start * (1 - anneal_progress_fraction)
            + config.gumbel_temperature_end * anneal_progress_fraction
        )
        history.gumbel_temperature.append(current_gumbel_temperature)

        last_loss_adv_value     = float('nan')
        last_loss_recon_value   = float('nan')
        last_loss_div_value     = float('nan')
        last_d_real_score       = float('nan')
        last_d_fake_before      = float('nan')
        last_d_fake_after       = float('nan')

        for clean_triple_indices_batch, target_triple_indices_batch in triple_pair_dataloader:
            clean_triple_indices_batch  = clean_triple_indices_batch.to(device)
            target_triple_indices_batch = target_triple_indices_batch.to(device)
            batch_size = clean_triple_indices_batch.size(0)

            real_label_tensor = torch.full(
                (batch_size, 1), config.real_label_value,
                device=device, dtype=torch.float,
            )
            fake_label_tensor = torch.full(
                (batch_size, 1), config.fake_label_value,
                device=device, dtype=torch.float,
            )

            # ─── D-STEP ───────────────────────────────────────────────
            discriminator_model.zero_grad()
            with torch.no_grad():
                clean_triple_embedding_for_d = build_triple_embedding(
                    clean_triple_indices_batch, entity_embedding, relation_embedding,
                )
                target_triple_embedding_for_d = build_triple_embedding(
                    target_triple_indices_batch, entity_embedding, relation_embedding,
                )
                latent_for_d = torch.randn(batch_size, config.latent_dim, device=device)
                gen_output_for_d = generator_model(
                    clean_triple_embedding_for_d, latent_for_d,
                    gumbel_temperature=current_gumbel_temperature,
                    return_soft_samples=True,
                )
                candidate_embedding_for_d = re_embed_soft_samples(
                    gen_output_for_d['head_entity_soft_sample'],
                    gen_output_for_d['relation_soft_sample'],
                    gen_output_for_d['tail_entity_soft_sample'],
                    entity_embedding.lookup_weight(),
                    relation_embedding.lookup_weight(),
                )

            clean_noised_real, target_noised_real = _add_instance_noise(
                clean_triple_embedding_for_d, target_triple_embedding_for_d,
                current_instance_noise_std,
            )
            d_score_on_real_pair = discriminator_model(clean_noised_real, target_noised_real)
            d_loss_real_pair     = binary_cross_entropy_with_logits_loss(
                d_score_on_real_pair, real_label_tensor,
            )

            clean_noised_fake, candidate_noised_fake = _add_instance_noise(
                clean_triple_embedding_for_d, candidate_embedding_for_d,
                current_instance_noise_std,
            )
            d_score_on_fake_pair = discriminator_model(clean_noised_fake, candidate_noised_fake)
            d_loss_fake_pair     = binary_cross_entropy_with_logits_loss(
                d_score_on_fake_pair, fake_label_tensor,
            )

            total_d_loss = d_loss_real_pair + d_loss_fake_pair
            total_d_loss.backward()
            last_d_real_score  = torch.sigmoid(d_score_on_real_pair).mean().item()
            last_d_fake_before = torch.sigmoid(d_score_on_fake_pair).mean().item()
            torch.nn.utils.clip_grad_norm_(
                discriminator_model.parameters(), max_norm=config.gradient_clip_max_norm,
            )
            discriminator_optimizer.step()
            epoch_d_loss_sum   += total_d_loss.item()
            epoch_d_loss_count += 1

            # ─── G-STEP ───────────────────────────────────────────────
            generator_optimizer.zero_grad()
            clean_triple_embedding_for_g = build_triple_embedding(
                clean_triple_indices_batch, entity_embedding, relation_embedding,
            )
            latent_sample_1 = torch.randn(batch_size, config.latent_dim, device=device)
            latent_sample_2 = torch.randn(batch_size, config.latent_dim, device=device)
            gen_output_z1 = generator_model(
                clean_triple_embedding_for_g, latent_sample_1,
                gumbel_temperature=current_gumbel_temperature, return_soft_samples=True,
            )
            gen_output_z2 = generator_model(
                clean_triple_embedding_for_g, latent_sample_2,
                gumbel_temperature=current_gumbel_temperature, return_soft_samples=True,
            )
            candidate_embedding_z1 = re_embed_soft_samples(
                gen_output_z1['head_entity_soft_sample'],
                gen_output_z1['relation_soft_sample'],
                gen_output_z1['tail_entity_soft_sample'],
                entity_embedding.lookup_weight(),
                relation_embedding.lookup_weight(),
            )
            candidate_embedding_z2 = re_embed_soft_samples(
                gen_output_z2['head_entity_soft_sample'],
                gen_output_z2['relation_soft_sample'],
                gen_output_z2['tail_entity_soft_sample'],
                entity_embedding.lookup_weight(),
                relation_embedding.lookup_weight(),
            )

            loss_recon_head = cross_entropy_loss(
                gen_output_z1['head_entity_logits'], target_triple_indices_batch[:, 0],
            )
            loss_recon_rel  = cross_entropy_loss(
                gen_output_z1['relation_logits'],    target_triple_indices_batch[:, 1],
            )
            loss_recon_tail = cross_entropy_loss(
                gen_output_z1['tail_entity_logits'], target_triple_indices_batch[:, 2],
            )
            loss_reconstruction = loss_recon_head + loss_recon_rel + loss_recon_tail

            clean_noised_g, candidate_noised_g = _add_instance_noise(
                clean_triple_embedding_for_g, candidate_embedding_z1,
                current_instance_noise_std,
            )
            d_score_on_fake_pair_g = discriminator_model(clean_noised_g, candidate_noised_g)
            loss_adversarial = binary_cross_entropy_with_logits_loss(
                d_score_on_fake_pair_g, real_label_tensor,
            )
            last_d_fake_after = torch.sigmoid(d_score_on_fake_pair_g).mean().item()

            loss_diversity = _mode_seeking_diversity_loss(
                candidate_embedding_z1, candidate_embedding_z2,
                latent_sample_1, latent_sample_2,
            )

            total_g_loss = (
                config.loss_weight_reconstruction * loss_reconstruction
                + config.loss_weight_adversarial  * loss_adversarial
                + config.loss_weight_diversity    * loss_diversity
            )
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                generator_parameter_list, max_norm=config.gradient_clip_max_norm,
            )
            generator_optimizer.step()

            last_loss_adv_value   = loss_adversarial.item()
            last_loss_recon_value = loss_reconstruction.item()
            last_loss_div_value   = loss_diversity.item()

            history.g_adversarial.append(last_loss_adv_value)
            history.g_reconstruction.append(last_loss_recon_value)
            history.g_diversity.append(last_loss_div_value)
            history.g_total.append(total_g_loss.item())
            history.d_total.append(total_d_loss.item())
            history.d_score_on_real.append(last_d_real_score)
            history.d_score_on_fake_before_g.append(last_d_fake_before)
            history.d_score_on_fake_after_g.append(last_d_fake_after)

        if (epoch_index + 1) % config.log_every_n_epochs == 0 or epoch_index == 0:
            print(
                f"[{epoch_index + 1:3d}/{config.total_epochs}]  "
                f"L_D: {epoch_d_loss_sum / max(epoch_d_loss_count, 1):.4f}  "
                f"L_G_adv: {last_loss_adv_value:.4f}  "
                f"L_G_recon: {last_loss_recon_value:.4f}  "
                f"L_G_div: {last_loss_div_value:+.4f}  "
                f"D(real): {last_d_real_score:.3f}  "
                f"D(fake): {last_d_fake_before:.3f}/{last_d_fake_after:.3f}  "
                f"noise_std: {current_instance_noise_std:.3f}  "
                f"gumbel_tau: {current_gumbel_temperature:.2f}",
                flush=True,
            )

        if (
            checkpoint_dir is not None
            and config.checkpoint_every_n_epochs > 0
            and (epoch_index + 1) % config.checkpoint_every_n_epochs == 0
            and (epoch_index + 1) < config.total_epochs
        ):
            save_checkpoint(
                checkpoint_dir / f"{checkpoint_prefix}_e{epoch_index + 1:04d}.pt",
                generator=generator_model,
                discriminator=discriminator_model,
                entity_embedding=entity_embedding,
                relation_embedding=relation_embedding,
                config=config,
                epoch=epoch_index + 1,
            )

    print("-" * 70, flush=True)
    print("Training complete.", flush=True)

    return {
        'generator':          generator_model,
        'discriminator':      discriminator_model,
        'entity_embedding':   entity_embedding,
        'relation_embedding': relation_embedding,
        'history':            history,
        'config':             config,
    }


def save_checkpoint(
    output_path: str | Path,
    *,
    generator: TripleGenerator,
    discriminator: TripleDiscriminator,
    entity_embedding: EntityEmbedding,
    relation_embedding: RelationEmbedding,
    config: TrainConfig,
    epoch: int,
    extra: Optional[Dict] = None,
) -> Path:
    """Save model state + config so sampling can stand alone.

    We deliberately do NOT serialise the optimisers - sampling doesn't need
    them and including them inflates the file 2-3x.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generator_state_dict':          generator.state_dict(),
        'discriminator_state_dict':      discriminator.state_dict(),
        'entity_embedding_state_dict':   entity_embedding.state_dict(),
        'relation_embedding_state_dict': relation_embedding.state_dict(),
        'config':                        config,
        'epoch':                         epoch,
        'num_entities':                  generator.num_entities,
        'num_relations':                 generator.num_relations,
    }
    if extra is not None:
        payload['extra'] = extra
    torch.save(payload, output_path)
    return output_path
