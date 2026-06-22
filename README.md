# SVD-Surgeon

**Optimal Singular-Value Surgery for LLM Compression**

SVD-Surgeon is a general corrective layer that improves SVD-based pruning methods. Given a truncated SVD of a weight matrix, SVD-Surgeon:

1. **Update:** Analytically corrects the retained singular values to compensate for the loss introduced by truncation, using a second-order (OBS) correction derived from the Fisher information.
2. **Selection:** Optionally selects which singular values to retain based on OBS sensitivity scores rather than magnitude alone.

SVD-Surgeon operates in a single shot with no iterative optimization or fine-tuning.

**Paper experiments:** For full reproduction of paper results (SVD-Surgeon applied on top of [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM)), see [our fork of SVD-LLM](https://github.com/mahmoud-safari/SVD-LLM/tree/svd_surgeon).

## Installation

```bash
git clone https://github.com/mahmoud-safari/SVD-Surgeon.git
cd svd-surgeon
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Plain SVD baseline for comparison
python compress.py --model facebook/opt-125m --ratio 0.9 --no_obs

# SVD-Surgeon compression
python compress.py --model facebook/opt-125m --ratio 0.9 --reuse_hbars --hbar_save_path hbar.pt

# With OBS-based selection
python compress.py --model facebook/opt-125m --ratio 0.9 --select_by_loss --reuse_hbars --hbar_save_path hbar.pt
```

On the first run with `--hbar_save_path`, SVD-Surgeon computes the spectral Fisher and saves it to `hbar.pt`. Subsequent runs with `--reuse_hbars` load it from disk, making compression much faster.


## License

MIT License
