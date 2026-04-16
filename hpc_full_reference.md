# IIT Delhi HPC (PADUM) — Full Technical Reference
# Scraped from https://supercomputing.iitd.ac.in on 2026-04-13

## Cluster Overview
- Name: PADUM
- Scheduler: PBS Pro v2022.1.3
- Contact: hpchelp@iitd.ac.in
- Acknowledgement: "The authors thank IIT Delhi HPC facility for computational resources."

---

## GPU Hardware (Relevant to NLP/DL)

| GPU | Node Type | VRAM | Queue/Flag | Notes |
|-----|-----------|------|------------|-------|
| NVIDIA K40 | Haswell | 12 GB | centos=haswell | CC 3.5; too old for modern PyTorch |
| NVIDIA V100 | Skylake | 32 GB | centos=skylake | CC 7.0; what Q1/Q2 are tested on |
| NVIDIA A100 40GB | IceLake | 40 GB | centos=icelake | CC 8.0 |
| NVIDIA A100 80GB | ScAI nodes | 80 GB | -q scai_q | 8x per node; best for large models |

ScAI nodes (scai01–scai04):
- scai01–scai03: 2x AMD EPYC 7282, 8x A100 80GB (scai01 PCIe; scai02–04 NVLink), 1TB RAM
- scai04: 2x Intel Xeon Platinum 8360Y, 4x A100 80GB, 512GB RAM

---

## Login

```bash
ssh username@hpc.iitd.ac.in          # general login
ssh username@gpu.hpc.iitd.ac.in      # GPU login nodes (klogin1/klogin2)
```
Outside IITD: use VPN first.

---

## Queues

| Queue | Notes |
|-------|-------|
| standard | Default; max 10 running + 10 queued per user |
| low | No budget required; max walltime 96h; cluster cap 2000 cores |
| high | Requires funded project + advanced HPC test; min 24 cores/6 GPUs |
| scai_q | ScAI A100 80GB nodes only |

---

## Job Submission — GPU Examples

### Interactive V100 (Skylake) — Q1/Q2 test environment
```bash
qsub -I -P <project> -q standard \
  -l select=1:ncpus=20:ngpus=1:centos=skylake \
  -l walltime=04:00:00
```

### Interactive A100 80GB (ScAI) — Q3 Llama inference
```bash
qsub -I -P <project> -q scai_q \
  -l select=1:ncpus=16:ngpus=1 \
  -l walltime=02:00:00
```

### Interactive A100 40GB (IceLake)
```bash
qsub -I -P <project> -q standard \
  -l select=1:ncpus=32:ngpus=1:centos=icelake \
  -l walltime=04:00:00
```

### Batch job template (GPU)
```bash
#!/bin/bash
#PBS -N job_name
#PBS -P <project_code>
#PBS -q standard
#PBS -l select=1:ncpus=20:ngpus=1:centos=skylake
#PBS -l walltime=04:00:00
#PBS -o output.log
#PBS -e error.log

cd $PBS_O_WORKDIR
module purge
module load compiler/gcc/11.2.0

# Activate teacher-provided vLLM env (Q3)
export PATH="/home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp/bin:$PATH"
conda activate /home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp
export LD_LIBRARY_PATH=/home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

python infer.py ...
```

---

## Conda in Batch Jobs

Conda `activate` often fails in batch scripts. Fix:
```bash
# Option 1: prepend env bin to PATH directly (used in our Q3 infer.sh)
export PATH="/path/to/env/bin:$PATH"

# Option 2: source conda init block from .bashrc into a separate file
source /full/path/to/condaBaseSetup
conda activate myenv
```

---

## PBS Management Commands

```bash
qstat -u $USER          # your jobs
qstat -f <jobID>        # full job info
qstat -saw <jobID>      # why job is not running
qstat -awT <jobID>      # estimated start time
qdel <jobID>            # delete job
qdel -W force <jobID>   # force delete
qstat -x                # recently finished jobs (last 24h)
tracejob -n 10 <jobID>  # job trace logs
```

---

## Storage

| Location | Size | Backed Up | Use |
|----------|------|-----------|-----|
| $HOME | 100 GB default | Yes | Code, checkpoints |
| $SCRATCH | 25 TB default | No | Active training data, large outputs |

- Submit jobs from `$SCRATCH` (faster I/O)
- Files in `$SCRATCH` older than 1 week may be deleted when disk fills

```bash
lfs quota -hu $USER /home     # check HOME quota
lfs quota -hu $USER /scratch  # check SCRATCH quota
```

---

## Internet Access from Compute Nodes

```bash
export SSL_CERT_FILE=$HOME/mycerts/CCIITD-CA.crt
lynx https://proxy61.iitd.ernet.in/cgi-bin/proxy.cgi   # PhD users use proxy61
# After login, note Proxy_IP and port (3128)
export http_proxy=<Proxy_IP>:3128
export https_proxy=<Proxy_IP>:3128
```

Proxy by user type: Faculty=proxy82, PhD=proxy61, MTech=proxy62, BTech/Staff=proxy21/22

Teacher confirmed: internet is available at evaluation time (no proxy needed for eval).

---

## File Transfer

```bash
scp -r localdir username@hpc.iitd.ac.in:~/targetdir
rsync -avz localdir/ username@hpc.iitd.ac.in:~/targetdir/
```

---

## Module System (Relevant Modules)

```bash
module load compiler/gcc/11.2.0           # GCC 11.2 (required before activating vLLM env)
module load apps/anaconda/3               # Anaconda3 (PyTorch 1.1.0 + TF 2.0, CUDA 10.0)
module load apps/pytorch/1.10.0/gpu/intelpython3.7  # PyTorch 1.10.0, CUDA 11.0
module load compiler/cuda/11.0/compilervars         # CUDA 11.0

module avail                              # list all modules
module purge                              # unload all
```

---

## Key Paths on HPC

```
Teacher vLLM env:   /home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp
Llama model:        /home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
Qwen model:         /home/scai/msr/aiy247541/scratch/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
CUDA runtime lib:   /home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
Sample scripts:     /home/apps/skeleton/
```

---

## Policies

- Do NOT run heavy jobs on login nodes (>50% CPU = account suspended for 24h)
- Do NOT use #PBS -V in job scripts
- Fix Windows line endings: `dos2unix script.sh`
- Job priority penalised -8 pts if <72 cores requested (GPU jobs exempt from this in practice)
- Standard queue: GPU cost = Rs. 1.00/hour/card
