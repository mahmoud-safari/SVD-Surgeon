"""
Compress an LLM using SVD-Surgeon (or plain SVD for comparison).

Usage:
    # SVD-Surgeon compression
    python compress.py --model facebook/opt-125m --ratio 0.9

    # Plain SVD baseline (no OBS correction)
    python compress.py --model facebook/opt-125m --ratio 0.9 --no_obs

    # With OBS selection enabled
    python compress.py --model facebook/opt-125m --ratio 0.9 --select_by_loss

For SVD-Surgeon applied on top of SVD-LLM, see the fork at:
    https://github.com/...

Requirements:
    pip install torch transformers datasets
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from svd_surgeon import compress_model


def get_calibration_data(dataset_name, tokenizer, nsamples, seqlen=2048, seed=3):
    """Load calibration data for Fisher estimation."""
    import random
    random.seed(seed)

    if dataset_name == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data["text"])
    elif dataset_name == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        text = "\n\n".join(data["text"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    calib_data = []
    for _ in range(nsamples + 10):  # request a few extra to handle short sequences
        i = random.randint(0, len(text) - seqlen * 10 - 1)
        enc = tokenizer(text[i : i + seqlen * 10], return_tensors="pt")
        if enc.input_ids.shape[1] < seqlen:
            continue
        inp = enc.input_ids[:, :seqlen]
        calib_data.append({
            "input_ids": inp,
            "attention_mask": torch.ones_like(inp),
        })
        if len(calib_data) >= nsamples:
            break

    print(f"Loaded {len(calib_data)} calibration samples from {dataset_name}")
    return calib_data


@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset_name="wikitext2", seqlen=2048, device="cuda"):
    """Evaluate perplexity on a test dataset."""
    if dataset_name == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    elif dataset_name == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = "\n\n".join(data[:1100]["text"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    nsamples = input_ids.numel() // seqlen
    nlls = []

    for i in range(nsamples):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen]
        output = model(batch)
        shift_logits = output.logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="SVD-Surgeon: OBS-corrected SVD compression for LLMs")

    # Model and data
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="HuggingFace model name or path")
    parser.add_argument("--ratio", type=float, default=0.9,
                        help="Compression ratio (fraction of parameters to keep)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4"])
    parser.add_argument("--device", type=str, default="cuda")

    # SVD-Surgeon hyperparameters
    parser.add_argument("--obs_scale", type=float, default=1.0,
                        help="OBS correction step size")
    parser.add_argument("--obs_damping", type=float, default=1e-5,
                        help="Regularization for H_SS solve")
    parser.add_argument("--obs_hdamping", type=float, default=1.0,
                        help="Regularization for selection scoring")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Candidate pool fraction beyond rank")
    parser.add_argument("--select_by_loss", action="store_true",
                        help="Use OBS sensitivity for selection")
    parser.add_argument("--no_obs", action="store_true",
                        help="Skip OBS correction (plain SVD baseline)")

    # Fisher estimation
    parser.add_argument("--obs_batches", type=int, default=8192,
                        help="Number of backward passes for Fisher estimation")
    parser.add_argument("--hbar_save_path", type=str, default=None,
                        help="Path to save/load precomputed Hbars")
    parser.add_argument("--reuse_hbars", action="store_true",
                        help="Load Hbars from hbar_save_path if available")

    # Evaluation
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip perplexity evaluation")
    parser.add_argument("--skip_original_eval", action="store_true",
                        help="Skip evaluation of the original (uncompressed) model")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu")

    # ------------------------------------------------------------------
    # Evaluate original model
    # ------------------------------------------------------------------
    if not args.skip_eval and not args.skip_original_eval:
        print("Evaluating original model...")
        model_eval = model.float().to(args.device)
        ppl_original = evaluate_ppl(
            model_eval, tokenizer, args.dataset, device=args.device)
        print(f"Original model PPL: {ppl_original:.2f}")
        model_eval = model_eval.cpu()
        del model_eval
        torch.cuda.empty_cache()
    else:
        ppl_original = None

    # ------------------------------------------------------------------
    # Load calibration data (only needed for SVD-Surgeon)
    # ------------------------------------------------------------------
    if args.no_obs:
        calib_data = None
    else:
        calib_data = get_calibration_data(
            args.dataset, tokenizer,
            nsamples=args.obs_batches,
            seqlen=2048)

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    model = model.float()
    method = "Plain SVD" if args.no_obs else "SVD-Surgeon"
    print(f"\nCompressing with {method} at ratio={args.ratio}...")

    compressed_model = compress_model(
        model,
        calib_data,
        ratio=args.ratio,
        obs_scale=args.obs_scale,
        obs_damping=args.obs_damping,
        obs_hdamping=args.obs_hdamping,
        alpha=args.alpha,
        select_by_loss=args.select_by_loss,
        no_obs=args.no_obs,
        obs_batches=args.obs_batches,
        device=args.device,
        hbar_save_path=args.hbar_save_path,
        reuse_hbars=args.reuse_hbars,
    )

    # ------------------------------------------------------------------
    # Evaluate compressed model
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print("\nEvaluating compressed model...")
        compressed_model = compressed_model.float().to(args.device)
        ppl_compressed = evaluate_ppl(
            compressed_model, tokenizer, args.dataset, device=args.device)
        print(f"\n{'='*50}")
        print(f"Method:          {method}")
        print(f"Model:           {args.model}")
        print(f"Ratio:           {args.ratio}")
        if ppl_original is not None:
            print(f"Original PPL:    {ppl_original:.2f}")
        print(f"Compressed PPL:  {ppl_compressed:.2f}")
        if ppl_original is not None:
            print(f"PPL increase:    {ppl_compressed - ppl_original:.2f} "
                  f"({(ppl_compressed / ppl_original - 1) * 100:.1f}%)")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
