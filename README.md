# VizFold Foundation

VizFold is a modular framework for running protein structure prediction models, extracting their internal representations (attention maps, hidden states, trunk intermediates), and visualizing them interactively. It produces a standardized **trace archive** that downstream tools can consume without knowing which model generated it.

---

## Supported Backends

| Backend | Model | Source | Status |
|---------|-------|--------|--------|
| **ESMFold** | `facebook/esmfold_v1` | HuggingFace Transformers | Fully integrated |
| **OpenFold** | AlphaFold2 weights | OpenFold (CUDA build) | [Legacy README](README_vizfold_openfold.md) |

---

## Quick Start (ESMFold)

### 1. Install

```bash
# Option A: conda (recommended)
conda env create -f environment-mac.yml   # or environment.yml on Linux
conda activate openfold-env

# Option B: pip (PyTorch must already be installed)
pip install -r requirements-esmfold.txt
```

### 2. Run inference

```bash
# Structure only
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode none

# Structure + attention + activations (GPU)
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --device cuda \
  --trace_mode attention+activations \
  --save_fp16
```

### 3. Explore outputs

```
outputs/esmf_6KWC/
├── meta.json                    # Run metadata, shapes, seed, model info
├── logs.txt
├── structure/
│   ├── predicted.pdb            # Full-atom PDB
│   └── predicted.pt             # Coordinate tensor
├── trace/
│   ├── index.json               # Layer → file path mapping
│   ├── summary.json             # Per-layer entropy, sparsity, norms
│   ├── attention/               # [B, H, N, N] per layer (36 layers)
│   ├── activations/             # [B, N, D] per layer
│   ├── trunk/                   # Evoformer intermediates
│   │   ├── block_000_seq.pt     # Per-block sequence state
│   │   ├── block_000_pair.pt    # Per-block pair state
│   │   ├── s_s.pt               # Final single representations
│   │   └── s_z.pt               # Final pair representations
│   └── structure_module/        # With --structure_traces
│       ├── ipa_attention/       # IPA attention [H, N, N] per recycle × block
│       └── backbone/            # Per-recycle coords and states
└── attention_files/             # VizFold .txt format (top-k per head)
```

See [docs/esmfold.md](docs/esmfold.md) for the full CLI reference, `meta.json` schema, and memory tips for long sequences.

---

## Interactive Dashboard

The frontend provides a browser-based visualization environment connecting to the ESMFold backend via a FastAPI bridge.

**Features:**
- **3D Structure Viewer** — WebGL rendering via 3Dmol.js with confidence (pLDDT) and spectrum coloring
- **Trace Explorer** — Plotly heatmaps for ESM-2 attention and trunk evolution (s_z) across recycling iterations
- **Bidirectional Residue Sync** — Click a residue in 3D to highlight it on the heatmap, or click a heatmap cell to highlight the residue in 3D
- **Timeline Controls** — Scrub through recycling iterations (0–3) or transformer layers (0–35)

```bash
# 1. Generate trace data
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out test_output \
  --trace_mode attention+activations

# 2. Start the API bridge
python server.py --dir test_output

# 3. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

See [frontend/README.md](frontend/README.md) for environment configuration and verification steps.

---

## HPC Deployment (PACE ICE)

ESMFold inference is tested on Georgia Tech's PACE ICE cluster with NVIDIA H100 GPUs via SLURM.

```bash
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

Key environment variables for HPC:

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | HuggingFace cache (avoid home quota) |
| `TORCH_HOME` | PyTorch hub cache |
| `TMPDIR` | Temp directory for large downloads |

See [docs/hpc_ice.md](docs/hpc_ice.md) for the full SLURM configuration and environment setup.

---

## Trace Extraction Architecture

ESMFold trace extraction uses PyTorch forward hooks to capture internal representations without modifying the model:

| Component | Hook Target | Output Shape |
|-----------|-------------|-------------|
| ESM-2 Attention | `esm.encoder.layer[i].attention.self` | `[B, H, N, N]` |
| ESM-2 Activations | `esm.encoder.layer[i]` | `[B, N, D]` |
| Trunk (s_s, s_z) | `trunk` (per recycle) | `[N, 1024]`, `[N, N, 128]` |
| IPA Attention | `trunk.structure_module` (patched softmax) | `[H, N, N]` |
| Backbone | `trunk.structure_module` output | `[num_blocks, B, N, 14, 3]` |

Special tokens (`<cls>`, `<eos>`) are stripped automatically so tensor dimensions align with the input sequence length.

---

## Tests

```bash
# Smoke tests (no GPU required)
python -m pytest tests/test_esmf_smoke.py -v

# Full validation (requires GPU and model download)
python -m pytest tests/test_esmf_smoke.py::test_esmfold_full_validation -v
```

---

## Project Structure

```
vizfold-foundation/
├── run_pretrained_esmf.py           # ESMFold CLI entrypoint
├── server.py                        # FastAPI bridge for frontend
├── vizfold/
│   └── backends/
│       ├── base.py                  # BackendBase interface
│       └── esmfold/
│           ├── hooks.py             # ESMFoldTraceCollector, StructureModuleTraceCollector
│           ├── inference.py         # ESMFoldRunner (implements BackendBase)
│           ├── schema.py            # meta.json builder
│           └── trace_adapter.py     # Archive writer (tensors, text, metadata)
├── frontend/                        # React + Vite dashboard
│   └── src/
│       ├── App.jsx                  # Layout and state management
│       └── components/
│           ├── StructureViewer.jsx   # 3Dmol.js panel
│           ├── TraceExplorer.jsx     # Plotly heatmap panel
│           └── TimelineControls.jsx  # Iteration/layer slider
├── tests/
│   └── test_esmf_smoke.py
├── scripts/hpc/ice/
│   └── run_esmf_ice.slurm
├── docs/
│   ├── esmfold.md
│   └── hpc_ice.md
└── examples/monomer/               # Sample FASTA inputs
```

---

## Reproducibility

- `--seed 42` fixes PyTorch RNG
- `--deterministic` enables CuDNN deterministic mode
- ESMFold inference is deterministic: repeated runs produce identical archives
- Both settings are recorded in `meta.json`

---

## Related Documentation

- [ESMFold Backend Reference](docs/esmfold.md) — Full CLI options, output schema, memory considerations
- [HPC / PACE ICE Guide](docs/hpc_ice.md) — SLURM scripts, environment setup, troubleshooting
- [Frontend Dashboard](frontend/README.md) — Setup, environment variables, verification
- [OpenFold Legacy Backend](README_vizfold_openfold.md) — Original OpenFold-based pipeline

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
See the [LICENSE](./LICENSE) file for details.
