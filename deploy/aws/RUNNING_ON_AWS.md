# Running the KGSAGE / ADKGD pipeline on AWS (g4dn.xlarge)

A single **g4dn.xlarge** GPU box that runs the training pipeline with **no
scheduler** — plain-bash runners, not SLURM. On one instance you own outright
there is nothing to schedule (one GPU, one job at a time), so the DeepThought
`.slurm` launchers are replaced here by two small scripts that carry the same
recipe and invocation:

| DeepThought | Here |
|---|---|
| `sbatch experiments/kgsage/slurm/train.slurm` | `bash deploy/aws/run.sh` |
| `sbatch experiments/slurm/exp_cell.slurm` | `bash deploy/aws/run_cell.sh` |

The tuned KGSAGE hyperparameters live as locked constants in
`kgsage/gan/train.py` (shared by both platforms) — `run.sh` only passes the
operational knobs (`DATASET`, `SEED`, `EPOCHS`, `SNAPSHOT_EVERY`).

## Hardware fit

| | g4dn.xlarge | Note |
|---|---|---|
| GPU | 1× **T4, 16 GB** (Turing 7.5) | on PyTorch 2.x's supported-arch list; comfortable for KGSAGE (dim=64) |
| vCPU / RAM | 4 / 16 GB | `run.sh` sets `OMP_NUM_THREADS=nproc` (=4) |
| local disk | 125 GB NVMe (ephemeral) | keep anything durable on the **EBS root** (or S3) |

KGSAGE `run.sh` sits well inside the T4. The heavier **ADKGD cells**
(`run_cell.sh`) also fit at the default batch, but if one OOMs the T4's 16 GB
VRAM, drop `MAX_EPOCH`/batch or use **g5.xlarge** (A10G, 24 GB, ~$0.44/hr spot).
Bigger g4dn sizes don't help — same single 16 GB T4.

## Cost

g4dn.xlarge Spot ≈ **$0.23/hr** → a full ~4–5 h run is **~$1**. Billing stops
only when the instance is stopped/terminated — not when the job ends.

---

## 0. Prerequisites (once)

- AWS CLI configured (`aws configure`) and an **EC2 key pair**.
- **GPU spot quota**: new accounts start at **0 vCPU** for "All G and VT Spot
  Instance Requests" — request **4 vCPU** in Service Quotas first (usually
  auto-approved fast).
- Push the `dev_gan_2` branch (with this `deploy/` kit) to origin — the launch
  auto-runs `bootstrap.sh` by cloning that branch:
  `git add deploy/ && git commit -m 'aws deploy kit' && git push origin dev_gan_2`
- Region: **ap-southeast-2 (Sydney)** = lowest latency from Flinders;
  **us-east-1** often cheaper/deeper T4 spot. Job is region-agnostic.

## 1. Launch — CloudFormation (recommended)

One stack = the spot instance + security group + IAM role, and `bootstrap.sh`
runs automatically on first boot. **`delete-stack` tears it all down in one
command** (no orphaned instance or spot request to forget about).

```bash
aws cloudformation deploy \
  --template-file deploy/aws/cfn-g4dn.yaml \
  --stack-name adkgd-kgsage \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides KeyName=YOUR_KEY YourIp=$(curl -s ifconfig.me)/32

aws cloudformation describe-stacks --stack-name adkgd-kgsage \
  --query 'Stacks[0].Outputs' --output table    # -> SSH command + public IP
```

<details><summary>Alternative: launch by hand (no CloudFormation)</summary>

```bash
AMI=$(aws ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text)
aws ec2 run-instances --image-id "$AMI" --instance-type g4dn.xlarge \
  --key-name YOUR_KEY --security-group-ids sg-XXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"spot"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=adkgd-kgsage}]' --count 1
# then SSH in and run bootstrap.sh manually (step 2).
```
</details>

## 2. Bootstrap the box

CloudFormation runs this for you; watch it with
`tail -f /var/log/cloud-init-output.log` after SSHing in. To run by hand (manual
launch, or to re-run):

```bash
curl -fsSL https://raw.githubusercontent.com/tharindu-wj/ADKGD/dev_gan_2/deploy/aws/bootstrap.sh | bash
exec bash -l          # fresh shell so conda lands on PATH
```

`bootstrap.sh` clones `dev_gan_2` to `~/ADKGD` (code + committed FB15K-237 /
WN18RR data), builds `~/envs/adkgd` (py3.11 + torch **cu121** + `torch_geometric`
+ sklearn), makes the runners executable, and verifies `torch.cuda.is_available()`
+ `RGCNConv` on the T4.

## 3. Run the pipeline

Work inside **tmux** so an SSH drop doesn't kill the run:

```bash
tmux new -s kgsage           # detach: Ctrl-b d ; reattach: tmux attach -t kgsage
conda activate ~/envs/adkgd
cd ~/ADKGD
```

**Train the KGSAGE generator** (per dataset / seed):

```bash
DATASET=fb15k237 bash deploy/aws/run.sh
DATASET=wn18rr   bash deploy/aws/run.sh
#   knobs: DATASET SEED EPOCHS SNAPSHOT_EVERY CKPT_PATH (same names as train.slurm)
DATASET=fb15k237 SEED=1 EPOCHS=8 SNAPSHOT_EVERY=1 bash deploy/aws/run.sh
```

Output: per-epoch snapshots `run_<tag>_s<seed>.epNN.pt` under
`experiments/kgsage/outputs/checkpoints/`, plus a `run_<tag>_s<seed>.out.txt`
(the corr-pick/alpha/dm-online training-health record — `run.sh` tees it for
you). Pick the best epoch afterwards:

```bash
python experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data data/FB15K-237
```

**Run ADKGD matrix cells** (the 2×2):

```bash
# baseline (rule-based negatives + rule-based test anomalies)
NEG_SOURCE=random TEST_SOURCE=random DATASET=FB15K-237 SEED=0 bash deploy/aws/run_cell.sh
# proposed (KGSAGE negatives + KGSAGE test anomalies) — needs a trained generator
NEG_SOURCE=gan TEST_SOURCE=gan DATASET=FB15K-237 SEED=0 \
  GAN_CKPT=experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.ep06.pt \
  bash deploy/aws/run_cell.sh

python experiments/aggregate_results.py --dataset FB15K-237
```

## 4. Monitor

```bash
tail -f experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.out.txt   # live training log
watch -n5 nvidia-smi                                                     # GPU util / VRAM
```

## 5. Get artifacts off the box (before you tear down)

```bash
# from your laptop:
scp -i YOUR_KEY.pem ubuntu@<ip>:~/ADKGD/experiments/kgsage/outputs/checkpoints/'*.pt' ./
scp -i YOUR_KEY.pem ubuntu@<ip>:~/ADKGD/experiments/kgsage/outputs/checkpoints/'*.out.txt' ./
```

Or sync to S3 from the box (attach an instance IAM role with `s3:PutObject` — the
CFN stack's role is a good place to add it):

```bash
./deploy/aws/sync_out.sh s3://your-bucket/adkgd
```

## 6. Stop billing

```bash
# CloudFormation: one command removes instance + SG + role + spot request
aws cloudformation delete-stack --stack-name adkgd-kgsage

# Manual launch: stop (keeps EBS) or terminate
aws ec2 stop-instances --instance-ids i-XXXX
aws ec2 terminate-instances --instance-ids i-XXXX
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/bin/bash^M: bad interpreter` | CRLF from a Windows clone. `sed -i 's/\r$//' deploy/aws/*.sh` (the `.gitattributes` here forces LF, so a fresh clone shouldn't hit this). |
| `run.sh: conda: command not found` | Run from a shell where conda is initialised (`exec bash -l` after bootstrap), or `source "$(conda info --base)/etc/profile.d/conda.sh"` first. |
| Preflight `no GPU` / `torch cannot see a GPU` | Wrong CUDA wheel vs driver. `nvidia-smi` for the driver's CUDA version, then reinstall torch with the matching wheel (`cu121` for the current DLAMI T4 driver). |
| `ImportError` on `RGCNConv` | PyG needs compiled extensions for your torch build: `TVER=$(python -c "import torch;print(torch.__version__.split('+')[0])"); pip install pyg-lib torch-scatter -f https://data.pyg.org/whl/torch-${TVER}+cu121.html` |
| CUDA OOM on an ADKGD cell (T4 16 GB) | Lower `MAX_EPOCH`/batch, or relaunch on **g5.xlarge** (A10G, 24 GB). All g4dn sizes share the same 16 GB T4. |
| Host OOM-kill (16 GB RAM) | WN18RR's sketch/support structures are heaviest; if killed, use **g4dn.2xlarge** (32 GB RAM, same T4). |
| ADKGD segfault very early | The OMP fix: `run_cell.sh` leaves `OMP_NUM_THREADS` unset so `run_experiment.py` setdefaults it to 1. If it still bites, `export OMP_NUM_THREADS=1` first. |
| Spot instance vanished mid-run | It was reclaimed. A ~$1 run is cheaper to rerun than to engineer resume — `sync_out.sh` to S3 first if you want the partial snapshots. With CloudFormation, re-`deploy` (spot interruption terminates the box). |
| CFN stack fails to create | Usually no spot capacity in that AZ/region (try another) or no default VPC (add a `SubnetId`). Read the failure on the stack's **Events** tab. |
