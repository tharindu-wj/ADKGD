"""Training playground: watch the GAN learn, layer by layer.

Run the whole file end-to-end:
    python experiments/gan/playground_train.py

Or step through cell-by-cell in VS Code / PyCharm / Jupyter - the `# %%`
markers are recognised as cell separators.

What you'll see:
  1. Load the dummy KG and look at real triples
  2. Build a Generator + Discriminator and inspect their embedding tables
  3. One Generator forward pass, with every tensor shape printed
  4. Gumbel-Softmax: why it lets us sample categorical and still backprop
  5. Soft-embedding: the "differentiable lookup" trick
  6. Discriminator forward pass
  7. Compute the actual training losses (recon + adv)
  8. One full D + G update step, before/after weights
  9. A 20-epoch mini-train, watch loss curves
 10. Save a small checkpoint for the test playground to load
"""
# %%
import os
import random
import sys

import torch
import torch.nn.functional as F

# Make the sibling modules importable without packaging ceremony.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_kg                                          # noqa: E402
from gan_model import Generator, Discriminator, gumbel_softmax, soft_embedding  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "data", "dummy_kg")
CKPT_OUT = os.path.join(os.path.dirname(__file__), "outputs", "checkpoints", "playground.pt")

random.seed(0)
torch.manual_seed(0)


# %%
# ----------------------------------------------------------------------
# 1. Load the knowledge graph
# ----------------------------------------------------------------------
print("=" * 70)
print("1) Loading the dummy KG")
print("=" * 70)

kg = load_kg(DATASET_DIR)

print(f"  entities = {kg['n_ent']:>4}    (e.g. {list(kg['ent2id'])[:4]} ...)")
print(f"  relations= {kg['n_rel']:>4}    (e.g. {list(kg['rel2id'])[:3]})")
print(f"  triples  = {len(kg['triples']):>4}    (unique = {len(kg['triple_set'])})")

print("\n  Three example real triples (strings):")
for h, r, t in kg["triples"][:3]:
    print(f"    ({kg['id2ent'][h]:<22} , {kg['id2rel'][r]:<18} , {kg['id2ent'][t]})")

print("\n  Same triples as integer IDs:")
for h, r, t in kg["triples"][:3]:
    print(f"    (h={h}, r={r}, t={t})")


# %%
# ----------------------------------------------------------------------
# 2. Build Generator and Discriminator (small enough to inspect)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) Build G and D")
print("=" * 70)

DIM = 16          # embedding size (small for readability; production uses 64-200)
Z_DIM = 8         # noise vector size
HIDDEN = 32       # hidden width of the generator MLP

G = Generator(kg["n_ent"], kg["n_rel"], dim=DIM, z_dim=Z_DIM, hidden=HIDDEN)
D = Discriminator(dim=DIM, hidden=32)

print(f"\n  Generator embedding tables:")
print(f"    ent_emb.weight shape = {tuple(G.ent_emb.weight.shape)}    ({kg['n_ent']} entities x {DIM} dims)")
print(f"    rel_emb.weight shape = {tuple(G.rel_emb.weight.shape)}    ({kg['n_rel']} relations x {DIM} dims)")

n_g_params = sum(p.numel() for p in G.parameters())
n_d_params = sum(p.numel() for p in D.parameters())
print(f"\n  Total parameters:")
print(f"    Generator    : {n_g_params:>6,}")
print(f"    Discriminator: {n_d_params:>6,}")

# Show the embedding for one entity. These vectors are LEARNED during training.
some_entity_id = 0
emb = G.ent_emb(torch.tensor(some_entity_id))
print(f"\n  Embedding for entity '{kg['id2ent'][some_entity_id]}' (id={some_entity_id}):")
print(f"    shape = {tuple(emb.shape)}")
print(f"    first 6 values = {emb.detach().numpy()[:6].round(3)}")


# %%
# ----------------------------------------------------------------------
# 3. Generator forward pass - every tensor, every shape
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) Generator forward pass on a batch of 2 real triples")
print("=" * 70)

# Build a tiny batch.
batch_triples = kg["triples"][:2]
h_in = torch.tensor([row[0] for row in batch_triples])
r_in = torch.tensor([row[1] for row in batch_triples])
t_in = torch.tensor([row[2] for row in batch_triples])
z = torch.randn(2, Z_DIM)

print(f"\n  Input integer IDs:")
print(f"    h : shape={tuple(h_in.shape)}    values={h_in.tolist()}    "
      f"(=> {[kg['id2ent'][i.item()] for i in h_in]})")
print(f"    r : shape={tuple(r_in.shape)}    values={r_in.tolist()}    "
      f"(=> {[kg['id2rel'][i.item()] for i in r_in]})")
print(f"    t : shape={tuple(t_in.shape)}    values={t_in.tolist()}    "
      f"(=> {[kg['id2ent'][i.item()] for i in t_in]})")
print(f"    z : shape={tuple(z.shape)}    (random noise vector - different each call)")

# Step A: look up embeddings
h_emb, r_emb, t_emb = G.lookup(h_in, r_in, t_in)
print(f"\n  After lookup() (Embedding tables map int -> vector):")
print(f"    h_emb : {tuple(h_emb.shape)}    r_emb : {tuple(r_emb.shape)}    t_emb : {tuple(t_emb.shape)}")

# Step B: concatenate + run the MLP
x = torch.cat([h_emb, r_emb, t_emb, z], dim=1)
print(f"\n  After concat([h_emb, r_emb, t_emb, z]):")
print(f"    x : {tuple(x.shape)}    (batch, 3*dim + z_dim = 3*{DIM} + {Z_DIM} = {3*DIM + Z_DIM})")

hidden = G.mlp(x)
print(f"\n  After 2-layer MLP (Linear -> ReLU -> Linear -> ReLU):")
print(f"    hidden : {tuple(hidden.shape)}    (batch, hidden_dim = {HIDDEN})")

# Step C: three output heads
head_logits = G.head_out(hidden)
rel_logits = G.rel_out(hidden)
tail_logits = G.tail_out(hidden)
print(f"\n  After three Linear heads (each projects 'hidden' to a vocabulary):")
print(f"    head_logits : {tuple(head_logits.shape)}    (batch, n_ent={kg['n_ent']})  "
      f"- a score for each possible head entity")
print(f"    rel_logits  : {tuple(rel_logits.shape)}    (batch, n_rel={kg['n_rel']})  "
      f"- a score for each possible relation")
print(f"    tail_logits : {tuple(tail_logits.shape)}    (batch, n_ent={kg['n_ent']})")

# What would a naive argmax pick?
print(f"\n  Argmax over head_logits for triple 0 (untrained model):")
predicted_head = head_logits[0].argmax().item()
print(f"    predicted head id = {predicted_head}  =>  '{kg['id2ent'][predicted_head]}'")
print(f"    real head was      = {h_in[0].item()}  =>  '{kg['id2ent'][h_in[0].item()]}'")
print(f"    (Untrained, so this is essentially random.)")


# %%
# ----------------------------------------------------------------------
# 4. Gumbel-Softmax: sample categorical AND keep gradients
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4) Gumbel-Softmax - why argmax breaks training, why Gumbel fixes it")
print("=" * 70)

# Toy: 5-class logits, see how three sampling methods differ.
toy_logits = torch.tensor([[2.0, 0.5, -1.0, 1.2, 0.0]])
print(f"\n  Raw logits (one row, 5 classes): {toy_logits[0].tolist()}")

print("\n  Three ways to pick a class:")
print(f"    plain argmax           : {toy_logits.argmax(dim=-1).item()}    "
      f"(deterministic - same answer every call, NO gradients)")

soft = torch.softmax(toy_logits, dim=-1)[0]
print(f"    softmax distribution    : {soft.detach().numpy().round(3).tolist()}    "
      f"(smooth - but argmax of this gives the same class every time)")

# Gumbel-Softmax: stochastic AND differentiable.
torch.manual_seed(1)
g1 = gumbel_softmax(toy_logits, tau=1.0)[0]
g2 = gumbel_softmax(toy_logits, tau=1.0)[0]
g3 = gumbel_softmax(toy_logits, tau=0.1)[0]  # low tau = near one-hot
print(f"    Gumbel-Softmax (tau=1.0): {g1.detach().numpy().round(3).tolist()}    sum={g1.sum().item():.3f}")
print(f"    Gumbel-Softmax (tau=1.0): {g2.detach().numpy().round(3).tolist()}    "
      f"(different draw - stochastic!)")
print(f"    Gumbel-Softmax (tau=0.1): {g3.detach().numpy().round(3).tolist()}    "
      f"(low temperature = near one-hot)")

# Importantly: Gumbel-Softmax outputs HAVE gradients. The vanilla argmax does not.
gumbel_out = gumbel_softmax(toy_logits, tau=1.0)
print(f"\n  Gumbel output requires gradient: {gumbel_out.requires_grad}     "
      f"(this is what lets the generator learn through sampling)")


# %%
# ----------------------------------------------------------------------
# 5. Soft-embedding: differentiable lookup
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("5) Soft-embedding = matrix multiply, which is differentiable")
print("=" * 70)

# Sample near-one-hot vectors for all three slots.
soft_h = gumbel_softmax(head_logits, tau=1.0)
soft_r = gumbel_softmax(rel_logits, tau=1.0)
soft_t = gumbel_softmax(tail_logits, tau=1.0)
print(f"\n  Gumbel samples:")
print(f"    soft_h : {tuple(soft_h.shape)}    (batch, n_ent)  - near-one-hot rows")
print(f"    soft_r : {tuple(soft_r.shape)}")
print(f"    soft_t : {tuple(soft_t.shape)}")

# The trick: soft_h @ ent_emb.weight is the same as ent_emb.weight[argmax(soft_h)]
# when soft_h is exactly one-hot - but it's smooth and differentiable.
candidate_emb = soft_embedding(soft_h, soft_r, soft_t, G.ent_emb.weight, G.rel_emb.weight)
print(f"\n  After soft_embedding (= 'differentiable lookup'):")
print(f"    candidate_emb : {tuple(candidate_emb.shape)}    (batch, 3, dim)")
print(f"    This embedding can be sent through the Discriminator AND backprop will reach G's params.")


# %%
# ----------------------------------------------------------------------
# 6. Discriminator: score the (real, candidate) pair
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("6) Discriminator forward pass")
print("=" * 70)

# Build the 'real' embedding (the actual input triple).
real_emb = torch.stack(list(G.lookup(h_in, r_in, t_in)), dim=1)
print(f"\n  real_emb (the input triple's embedding) : {tuple(real_emb.shape)}")
print(f"  candidate_emb (G's output, fake)        : {tuple(candidate_emb.shape)}")

# D scores the pair (real, candidate) - high = "candidate looks real", low = "looks fake".
score = D(real_emb, candidate_emb)
print(f"\n  Discriminator score : {tuple(score.shape)}    raw logits = {score.detach().numpy().flatten().round(3).tolist()}")
print(f"  As probabilities    : {torch.sigmoid(score).detach().numpy().flatten().round(3).tolist()}    "
      f"(0 = clearly fake, 1 = clearly real)")
print(f"  Untrained D is near 0.5 - it can't tell yet.")


# %%
# ----------------------------------------------------------------------
# 7. Compute the actual training losses
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("7) Losses - what G and D are trying to optimise")
print("=" * 70)

# For the reconstruction loss we need a "target" - a random slot-corruption
# of the real triple. The generator should learn to produce this distribution.
def random_corrupt(h, r, t, n_ent, n_rel):
    slot = random.randint(0, 2)
    if slot == 0:
        return (random.randint(0, n_ent - 1), r, t)
    if slot == 1:
        return (h, random.randint(0, n_rel - 1), t)
    return (h, r, random.randint(0, n_ent - 1))

targets = [random_corrupt(h, r, t, kg["n_ent"], kg["n_rel"]) for h, r, t in batch_triples]
target_h = torch.tensor([t[0] for t in targets])
target_r = torch.tensor([t[1] for t in targets])
target_t = torch.tensor([t[2] for t in targets])
print(f"\n  Training pair 0:")
print(f"    real   : ({kg['id2ent'][h_in[0].item()]}, {kg['id2rel'][r_in[0].item()]}, {kg['id2ent'][t_in[0].item()]})")
print(f"    target : ({kg['id2ent'][target_h[0].item()]}, {kg['id2rel'][target_r[0].item()]}, {kg['id2ent'][target_t[0].item()]})")

# Reconstruction loss (cross entropy across the 3 vocabs).
loss_recon = (
    F.cross_entropy(head_logits, target_h)
    + F.cross_entropy(rel_logits, target_r)
    + F.cross_entropy(tail_logits, target_t)
)
print(f"\n  Reconstruction loss = CE(head) + CE(rel) + CE(tail) = {loss_recon.item():.4f}")
print(f"    'How well do G's logits predict the random-corrupt target?'")

# Adversarial loss (G wants D to think the candidate is real, i.e. label=1).
score_for_g = D(real_emb, candidate_emb)
loss_adv = F.binary_cross_entropy_with_logits(
    score_for_g, torch.ones_like(score_for_g),
)
print(f"\n  Adversarial loss    = BCE(D's score on candidate, label=1) = {loss_adv.item():.4f}")
print(f"    'How badly is G failing to fool D?'")

# Total G loss = adv + 10*recon (Pix2Pix-style weighting).
loss_g = loss_adv + 10.0 * loss_recon
print(f"\n  Total G loss        = adv + 10 * recon = {loss_g.item():.4f}")


# %%
# ----------------------------------------------------------------------
# 8. One training step: parameters before vs after
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("8) One training step - D update, then G update")
print("=" * 70)

opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
opt_D = torch.optim.Adam(D.parameters(), lr=2.5e-4)

# Snapshot one weight before training.
g_first_layer_weight_before = G.mlp[0].weight.data.clone()
print(f"\n  G.mlp[0].weight[0,:6] BEFORE  = {g_first_layer_weight_before[0, :6].numpy().round(4)}")

# --- D step ---
with torch.no_grad():
    fresh_logits = G(h_in, r_in, t_in, torch.randn(2, Z_DIM))
    soft = [gumbel_softmax(l, tau=1.0) for l in fresh_logits]
    cand_emb = soft_embedding(*soft, G.ent_emb.weight, G.rel_emb.weight)
    real_emb_for_d = torch.stack(list(G.lookup(h_in, r_in, t_in)), dim=1)
    target_emb_for_d = torch.stack(list(G.lookup(target_h, target_r, target_t)), dim=1)

opt_D.zero_grad()
sc_target = D(real_emb_for_d, target_emb_for_d)
sc_fake = D(real_emb_for_d, cand_emb)
loss_d = (
    F.binary_cross_entropy_with_logits(sc_target, torch.ones_like(sc_target))
    + F.binary_cross_entropy_with_logits(sc_fake, torch.zeros_like(sc_fake))
)
loss_d.backward()
opt_D.step()
print(f"  D step    : loss = {loss_d.item():.4f}    "
      f"D(target)={torch.sigmoid(sc_target).mean().item():.3f}    "
      f"D(fake)={torch.sigmoid(sc_fake).mean().item():.3f}")

# --- G step ---
opt_G.zero_grad()
logits = G(h_in, r_in, t_in, torch.randn(2, Z_DIM))
soft = [gumbel_softmax(l, tau=1.0) for l in logits]
cand_emb = soft_embedding(*soft, G.ent_emb.weight, G.rel_emb.weight)
real_emb_for_g = torch.stack(list(G.lookup(h_in, r_in, t_in)), dim=1)

sc_fake_for_g = D(real_emb_for_g, cand_emb)
l_adv = F.binary_cross_entropy_with_logits(sc_fake_for_g, torch.ones_like(sc_fake_for_g))
l_recon = (
    F.cross_entropy(logits[0], target_h)
    + F.cross_entropy(logits[1], target_r)
    + F.cross_entropy(logits[2], target_t)
)
loss_g = l_adv + 10.0 * l_recon
loss_g.backward()
opt_G.step()
print(f"  G step    : loss = {loss_g.item():.4f}    "
      f"adv={l_adv.item():.4f}    recon={l_recon.item():.4f}")

# Snapshot after - see that the weight actually moved.
g_first_layer_weight_after = G.mlp[0].weight.data
delta = (g_first_layer_weight_after - g_first_layer_weight_before).abs().mean().item()
print(f"\n  G.mlp[0].weight[0,:6] AFTER   = {g_first_layer_weight_after[0, :6].numpy().round(4)}")
print(f"  Mean abs change across that layer = {delta:.6f}  (small but nonzero = learning)")


# %%
# ----------------------------------------------------------------------
# 9. Mini-training: 20 epochs on dummy_kg, watch loss curves
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("9) Mini-train (20 epochs on dummy_kg, full dataset)")
print("=" * 70)

# Re-seed and re-init so this section is self-contained if you skip earlier ones.
torch.manual_seed(0)
random.seed(0)

G = Generator(kg["n_ent"], kg["n_rel"], dim=DIM, z_dim=Z_DIM, hidden=HIDDEN)
D = Discriminator(dim=DIM, hidden=32)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2.5e-4, betas=(0.5, 0.999))

# Build training pairs once.
pairs = [
    ((h, r, t), random_corrupt(h, r, t, kg["n_ent"], kg["n_rel"]))
    for h, r, t in kg["triples"]
]

print(f"\n  {len(pairs)} training pairs, batch_size = 32")
print("  epoch  D_loss   G_loss   adv     recon")
print("  -----  -------  -------  ------  ------")

for epoch in range(1, 21):
    random.shuffle(pairs)
    epoch_d, epoch_g, epoch_adv, epoch_rec, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    for start in range(0, len(pairs), 32):
        batch = pairs[start:start + 32]
        if len(batch) < 2:
            continue

        real_t = torch.tensor([p[0] for p in batch])
        targ_t = torch.tensor([p[1] for p in batch])
        h_b, r_b, t_b = real_t[:, 0], real_t[:, 1], real_t[:, 2]
        th_b, tr_b, tt_b = targ_t[:, 0], targ_t[:, 1], targ_t[:, 2]

        # D step
        with torch.no_grad():
            l = G(h_b, r_b, t_b, torch.randn(len(batch), Z_DIM))
            s = [gumbel_softmax(li, tau=1.0) for li in l]
            cand_e = soft_embedding(*s, G.ent_emb.weight, G.rel_emb.weight)
            real_e = torch.stack(list(G.lookup(h_b, r_b, t_b)), dim=1)
            targ_e = torch.stack(list(G.lookup(th_b, tr_b, tt_b)), dim=1)

        opt_D.zero_grad()
        sc_t = D(real_e, targ_e)
        sc_f = D(real_e, cand_e)
        l_d = (
            F.binary_cross_entropy_with_logits(sc_t, torch.ones_like(sc_t))
            + F.binary_cross_entropy_with_logits(sc_f, torch.zeros_like(sc_f))
        )
        l_d.backward()
        opt_D.step()

        # G step
        opt_G.zero_grad()
        l = G(h_b, r_b, t_b, torch.randn(len(batch), Z_DIM))
        s = [gumbel_softmax(li, tau=1.0) for li in l]
        cand_e = soft_embedding(*s, G.ent_emb.weight, G.rel_emb.weight)
        real_e = torch.stack(list(G.lookup(h_b, r_b, t_b)), dim=1)

        sc = D(real_e, cand_e)
        l_a = F.binary_cross_entropy_with_logits(sc, torch.ones_like(sc))
        l_r = F.cross_entropy(l[0], th_b) + F.cross_entropy(l[1], tr_b) + F.cross_entropy(l[2], tt_b)
        l_g = l_a + 10.0 * l_r
        l_g.backward()
        opt_G.step()

        epoch_d += l_d.item()
        epoch_g += l_g.item()
        epoch_adv += l_a.item()
        epoch_rec += l_r.item()
        n_batches += 1

    if epoch == 1 or epoch % 2 == 0 or epoch == 20:
        print(f"  {epoch:>5}  {epoch_d/n_batches:>7.4f}  {epoch_g/n_batches:>7.4f}  "
              f"{epoch_adv/n_batches:>6.4f}  {epoch_rec/n_batches:>6.4f}")

print("\n  Things to look for in those numbers:")
print("    - G_loss should generally trend DOWN  (G is getting better at fooling D)")
print("    - D_loss should stay near ln(2)*2 ~ 1.39  (D struggles when G improves)")
print("    - If D_loss falls below 0.5 fast: D is winning -> consider spectral_norm")
print("    - If G_loss plateaus high: G is stuck -> check recon_weight or LR")


# %%
# ----------------------------------------------------------------------
# 10. Save a small checkpoint for the test playground
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("10) Save a checkpoint")
print("=" * 70)

os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
torch.save({
    "generator_state": G.state_dict(),
    "ent2id": kg["ent2id"],
    "rel2id": kg["rel2id"],
    "id2ent": kg["id2ent"],
    "id2rel": kg["id2rel"],
    "real_triples": list(kg["triple_set"]),
    "n_ent": kg["n_ent"],
    "n_rel": kg["n_rel"],
    "dim": DIM,
    "z_dim": Z_DIM,
    "hidden": HIDDEN,
}, CKPT_OUT)
print(f"\n  Saved to: {CKPT_OUT}")
print(f"  Now run:   python experiments/gan/playground_test.py")

# %%
