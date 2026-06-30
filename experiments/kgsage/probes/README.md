# KGSAGE probes — understand the pipeline one rung at a time

A learning ladder. Each script makes **one** part of the pipeline produce
something you can *read*, instead of an aggregate metric. Run them in order.

```bash
export PYTHONPATH=experiments          # so `import kgsage` resolves
python experiments/kgsage/probes/rung1_data.py
python experiments/kgsage/probes/rung2_antisymmetry.py
...
```

Every probe **auto-detects** the dataset + checkpoints:
- uses `data/FB15K-237` if present, else falls back to `data/dummy_kg`;
- override the dataset by passing a directory: `... rung1_data.py data/WN18RR`.

| Rung | Question it answers | Runs where |
|---|---|---|
| 1 `rung1_data.py` | What is the model looking at? (triples, vocab, a neighborhood) | 🟢 local |
| 2 `rung2_antisymmetry.py` | What makes a role-swap a contradiction? (count reverses) | 🟢 local |
| 3 `rung3_encoder_predict.py` | Is the encoder good at prediction? (top-5 tails, real-vs-fake) | 🔴 HPC (PyG + encoder ckpt) |
| 4 `rung4_embeddings.py` | What did the embeddings learn? (4a relations 🟢-ish, 4b entities 🔴) | mixed |
| 5 `rung5_supervision.py` | What does the GAN learn *from*? (real 2-cycles vs templates) | 🟢 local |
| 6 `rung6_generator.py` | What contradiction does the GAN propose? (+ symmetric rejection) | 🟢 local |
| 7 `rung7_trace.py` | One triple through every rung | 🟢/🔴 |

🟢 = runs on a laptop (CPU, no `torch_geometric`).
🔴 = needs the encoder forward → `torch_geometric` + `fb15k237_encoder.pt` (the HPC).
Rungs that can't run print a clear `[skip]` with what to do, instead of crashing.

**Suggested path**
- *Locally, today:* rungs **1, 2, 5, 6, 7** — covers anti-symmetry → supervision →
  generation, the conceptual spine, using the checkpoints you already have.
- *On HPC (next encoder session):* rungs **3** and **4** to watch the encoder predict.

Each script ends with a `>> LOOK FOR:` line telling you what the output means.
