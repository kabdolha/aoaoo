[Original Article]()

## Appendix A: Definitions

- Cluster - One or more interconnected virtual machines that connect over high speed networking in order to compute large tasks together.
- Device - A single virtual machine or server that has accesses to 1 or more accelerators.
- Accelerator - A single accelerator chip, has accesses to 1 or more cores. This can be a GPU, Trainium Chip, Inferentia Chip, TPU, etc
- Core - A single computational unit, this unit takes in data from memory, performs a calculation, and writes back to memory.

## Appendix B: Bonus Example

Let's walk from infrastructure down to the core, and outline exactly what commands or APIs are run at each level to make something like `all_gather` actually happen on a distributed system with PyTorch + Slurm on AWS.

We’ll assume:

- 2 EC2 instances (`p4d.24xlarge` each)
- 8 GPUs per instance (NVIDIA A100)
- Using Slurm to launch
- Using PyTorch with DDP (`DistributedDataParallel`)
- Collective op is `all_gather`

---

### 1. Infrastructure Layer (Slurm Job Submission)

This layer handles job scheduling and launching processes.

**Command Run:**

```bash
srun --ntasks=16 \
--ntasks-per-node=8 \
--gpus-per-task=1 \
--cpus-per-task=6 \
--nodes=2 \
--exclusive \
--output=out.txt \
python train.py
```

**What Happens:**

- Slurm allocates 2 nodes, 8 tasks per node = 16 total tasks (1 per GPU)
- One Python process is launched per GPU
- It sets environment variables like:
  - `RANK`, `WORLD_SIZE`, `SLURM_NODEID`, `CUDA_VISIBLE_DEVICES`
  - `MASTER_ADDR`, `MASTER_PORT` (used for rendezvous)

---

### 2. Device Layer (Node/Host Setup)

This is per-node orchestration and GPU visibility setup.

**Environment Setup Inside Each Node (implicit via Slurm):**

```bash
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
```

Each process:

- Knows its global rank
- Selects its local GPU using `LOCAL_RANK`:

```python
torch.cuda.set_device(local_rank)
```

---

### 3. Accelerator Layer (Inter-GPU Communication)

Now, each process attaches to a GPU and sets up NCCL communication (either intra-node or inter-node).

**PyTorch Code (from `train.py`):**

```python
import torch
import torch.distributed as dist

dist.init_process_group(
backend='nccl',
init_method='env://', # uses the env vars set above
)

model = MyModel().cuda()
model = torch.nn.parallel.DistributedDataParallel(
model, device_ids=[local_rank]
)
```

NCCL does the following:

- **Within each node:**
  - Talks to all other processes using NVLink
- **Across nodes:**
  - Uses EFA or TCP/IP (depending on EC2 instance and config)
  - Establishes NCCL rings or trees for collectives
  - Communicates via RDMA or TCP socket

---

### 4. Core Layer (The `all_gather` Execution)

This is where the actual collective op happens.

**Example in PyTorch:**

```python

# Each process sends its tensor to all other processes

tensor = torch.tensor([rank], device='cuda')
gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

torch.distributed.all_gather(gathered, tensor)
```

NCCL now:

- Does intra-node communication first (fast NVLink)
- Then inter-node communication (slower, via EFA/TCP)
- Ensures every process ends up with the full list

---

### Summary:

| Layer       | Action                                   | Example Command/API                                        |
| ----------- | ---------------------------------------- | ---------------------------------------------------------- |
| Infra       | Job scheduling and launch                | `srun --ntasks=16 ... python train.py`                     |
| Device      | Env setup, rank mapping, GPU selection   | `export RANK`, `torch.cuda.set_device(local_rank)`         |
| Accelerator | DDP init, NCCL backend setup             | `init_process_group(backend='nccl', init_method='env://')` |
| Core        | Execute `all_gather` or other collective | `torch.distributed.all_gather(...)`                        |
