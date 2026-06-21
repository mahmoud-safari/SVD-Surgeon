"""
SVD-Surgeon: Optimal Brain Surgeon for SVD-based Network Compression

A general corrective layer that improves any SVD-based pruning method by:
  1. Analytically correcting retained singular values to compensate for
     truncation loss (update step).
  2. Selecting which singular values to retain based on OBS sensitivity
     scores rather than magnitude (selection step).

Usage:
    from svd_surgeon import compress_model

    compressed_model = compress_model(
        model,
        calib_data,          # list of dicts with "input_ids" and "attention_mask"
        ratio=0.5,           # fraction of parameters to keep
        obs_scale=1.0,       # OBS correction step size
        obs_damping=1e-5,    # damping for H_SS regularization
        obs_hdamping=1.0,    # damping for selection scoring
        alpha=0.3,           # candidate pool fraction
        select_by_loss=False, # enable OBS-based selection
        obs_batches=8192,    # number of backward passes for Fisher estimation
        device="cuda",
    )

Reference:
    SVD-Surgeon: Optimal Singular-Value Surgery for LLM Compression
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path


# ============================================================
# Core OBS Functions
# ============================================================

def spectral_grad_from_weight_grad(
    G: torch.Tensor,
    U: torch.Tensor,
    Vh: torch.Tensor,
    full_k: int,
) -> torch.Tensor:
    """
    Project a weight-space gradient G (out_dim, in_dim) onto the SVD
    basis to obtain a spectral gradient vector of length full_k.

    The spectral gradient is:  gbar_i = u_i^T G v_i  for i = 1, ..., full_k

    Args:
        G:      Weight gradient tensor of shape (out_dim, in_dim).
        U:      Left singular vectors, shape (out_dim, full_k).
        Vh:     Right singular vectors (transposed), shape (full_k, in_dim).
        full_k: Number of singular value directions to project onto.

    Returns:
        gbar: Spectral gradient vector of shape (full_k,).
    """
    device = G.device
    U_k = U[:, :full_k].to(device)
    V_k = Vh[:full_k, :].T.to(device)
    gbar = (U_k * (G.float() @ V_k)).sum(dim=0)
    return gbar


def obs_select_by_score(
    S: torch.Tensor,
    Hbar: torch.Tensor,
    rank: int,
    full_k: int,
    damping: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select which singular values to keep based on OBS sensitivity scores.

    The OBS saliency for dropping singular value i is:
        score_i = sigma_i^2 / [H^{-1}]_{ii}

    Higher scores indicate more important singular values.  We keep
    the top-rank indices by score.

    Args:
        S:       Singular values, shape (full_k,).
        Hbar:    Spectral Fisher matrix, shape (full_k, full_k).
        rank:    Number of singular values to keep.
        full_k:  Total candidate pool size.
        damping: Regularization coefficient (relative to diagonal mean).

    Returns:
        keep_idx: Sorted indices of kept singular values.
        drop_idx: Indices of dropped singular values.
    """
    device = S.device
    dtype = S.dtype
    H = Hbar.to(device=device, dtype=dtype)
    dmean = torch.diag(H).mean().clamp(min=1e-12)
    H = H + damping * dmean * torch.eye(full_k, device=device, dtype=dtype)
    scores = S[:full_k].pow(2) / torch.diag(torch.linalg.pinv(H))
    keep_idx, _ = torch.sort(torch.topk(scores, rank).indices)
    mask = torch.ones(full_k, dtype=torch.bool, device=device)
    mask[keep_idx] = False
    return keep_idx, torch.arange(full_k, device=device)[mask]


def obs_correct_singular_values(
    S: torch.Tensor,
    rank: int,
    Hbar: torch.Tensor,
    obs_scale: float = 1.0,
    obs_damping: float = 1e-5,
    obs_hdamping: float = 1.0,
    alpha: float = 0.3,
    select_by_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    OBS correction in the SVD basis.

    Given singular values S, the target rank, and the spectral Fisher
    Hbar, this function:

      1. Determines the candidate pool: full_k = rank + alpha * (total - rank).
      2. Optionally selects which rank indices to keep by OBS
         sensitivity (select_by_loss=True) rather than by magnitude.
      3. Solves for the correction delta that compensates for the
         dropped singular values:
             delta = H_SS^{-1} H_SC sigma_C
      4. Returns the corrected kept singular values.

    Args:
        S:              Full singular value vector (descending order).
        rank:           Number of singular values to keep.
        Hbar:           Spectral Fisher matrix, shape >= (full_k, full_k).
        obs_scale:      Step-size scalar for the correction delta.
        obs_damping:    Diagonal regularization for H_SS solve.
        obs_hdamping:   Diagonal regularization for selection scoring.
        alpha:          Candidate pool fraction: full_k = rank + alpha*(total-rank).
        select_by_loss: If True, select kept indices by OBS sensitivity.

    Returns:
        sigma_corrected: Corrected singular values for kept indices, shape (rank,).
        keep_idx:        Indices of kept singular values within [0, full_k).
        drop_idx:        Indices of dropped singular values within [0, full_k).
    """
    device = S.device
    rank = min(rank, S.numel())
    full_k = rank + int((S.numel() - rank) * alpha)

    S_block = S[:full_k]
    Hbar = Hbar[:full_k, :full_k].to(device=device, dtype=S.dtype)

    # --- Step 1: Select which singular values to keep ---
    if select_by_loss:
        keep_idx, drop_idx = obs_select_by_score(
            S_block, Hbar, rank, full_k, obs_hdamping)
    else:
        keep_idx = torch.arange(rank, device=device)
        drop_idx = torch.arange(rank, full_k, device=device)

    sigma_S = S_block[keep_idx]
    sigma_C = S_block[drop_idx]

    if sigma_C.numel() == 0:
        return sigma_S, keep_idx, drop_idx

    # --- Step 2: Compute OBS correction ---
    H_SS = Hbar[keep_idx][:, keep_idx]
    H_SC = Hbar[keep_idx][:, drop_idx]

    dmean = torch.diag(H_SS).mean().clamp(min=1e-12)
    H_SS = H_SS + obs_damping * dmean * torch.eye(
        H_SS.shape[0], device=device, dtype=H_SS.dtype)

    delta = torch.linalg.solve(H_SS, H_SC @ sigma_C)

    return torch.clamp(sigma_S + obs_scale * delta, min=0.0), keep_idx, drop_idx


# ============================================================
# Fisher (Hbar) Collection
# ============================================================

def collect_spectral_fisher(
    model: nn.Module,
    calib_data: list,
    layer_svd_info: Dict,
    num_batches: int = 8192,
    device: str = "cuda",
    hbar_save_path: Optional[str] = None,
    reuse_hbars: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Accumulate the spectral Fisher (Hbar) for every linear layer.

    For each calibration batch, runs a forward + backward pass and
    projects the weight gradient of each layer onto its SVD basis
    to accumulate the outer product  gbar @ gbar^T.

    Uses backward hooks for memory efficiency (gradients are freed
    immediately after projection), and gradient checkpointing for
    models with > 1B parameters.

    Args:
        model:          The model to collect Fisher information from.
        calib_data:     List of calibration batches (dicts with
                        "input_ids" and "attention_mask").
        layer_svd_info: Dict mapping layer name -> {"U", "Vh", "full_k", ...}.
        num_batches:    Number of backward passes.
        device:         CUDA device string.
        hbar_save_path: Optional path to save/load Hbars.
        reuse_hbars:    If True and path exists, load cached Hbars.

    Returns:
        Hbars: Dict mapping layer name -> Hbar tensor (full_k, full_k).
    """
    # --- Load cached Hbars if requested ---
    if reuse_hbars and hbar_save_path is not None:
        path = Path(hbar_save_path)
        if path.exists():
            print(f"[Hbar] Loading cached Hbars from {path}")
            return torch.load(path, map_location=device)
        else:
            print(f"[Hbar] reuse_hbars=True but {path} not found — recomputing.")

    # --- Initialize Hbars on CPU ---
    Hbars = {
        n: torch.zeros(info["full_k"], info["full_k"], device="cpu")
        for n, info in layer_svd_info.items()
    }

    # --- Setup model ---
    model_params = sum(p.numel() for p in model.parameters())
    use_grad_ckpt = model_params > 1e9

    model = model.to(device)
    if use_grad_ckpt:
        model = model.half()
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled ({model_params / 1e9:.1f}B params)")
    model.train()
    print(f"Model on device: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # --- Register backward hooks ---
    hooks = []
    for name, info in layer_svd_info.items():
        module = model
        for part in name.split("."):
            module = getattr(module, part)

        def make_hook(n, inf):
            def hook(grad):
                grad_float = grad.float()
                U_gpu = inf["U"].to(grad.device)
                Vh_gpu = inf["Vh"].to(grad.device)
                gbar = spectral_grad_from_weight_grad(
                    grad_float.detach(), U_gpu, Vh_gpu, inf["full_k"])
                Hbars[n] += torch.outer(gbar, gbar).cpu()
                del gbar, grad_float, U_gpu, Vh_gpu
                return None
            return hook

        hooks.append(module.weight.register_hook(make_hook(name, info)))

    # --- Accumulate Fisher ---
    for i, batch in enumerate(calib_data):
        if i >= num_batches:
            break
        if i % 500 == 0:
            print(f"Fisher batch {i}/{num_batches}")
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        del loss
        for k in batch:
            batch[k] = batch[k].cpu()
        torch.cuda.empty_cache()

    # --- Cleanup ---
    for h in hooks:
        h.remove()
    if use_grad_ckpt:
        model.gradient_checkpointing_disable()
    model.eval()
    model = model.cpu().float()
    torch.cuda.empty_cache()

    # --- Save if requested ---
    if hbar_save_path is not None:
        path = Path(hbar_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(Hbars, path)
        print(f"[Hbar] Saved Hbars to {path}")

    return Hbars


# ============================================================
# Model Compression
# ============================================================

def _find_linear_layers(module: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """Recursively find all nn.Linear layers in a module."""
    layers = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            layers[full_name] = child
        else:
            layers.update(_find_linear_layers(child, full_name))
    return layers


def _compute_rank(weight_shape: torch.Size, ratio: float) -> int:
    """Compute target rank given weight shape and compression ratio."""
    m, n = weight_shape
    rank = int(m * n * ratio / (m + n))
    return max(1, rank)


def compress_model(
    model: nn.Module,
    calib_data: Optional[list] = None,
    ratio: float = 0.5,
    obs_scale: float = 1.0,
    obs_damping: float = 1e-5,
    obs_hdamping: float = 1.0,
    alpha: float = 0.3,
    select_by_loss: bool = False,
    no_obs: bool = False,
    obs_batches: int = 8192,
    device: str = "cuda",
    hbar_save_path: Optional[str] = None,
    reuse_hbars: bool = False,
    target_layers: Optional[List[str]] = None,
) -> nn.Module:
    """
    Compress a model using SVD truncation, optionally with OBS correction.

    When no_obs=False (default), applies SVD-Surgeon: vanilla SVD followed
    by OBS correction of retained singular values and optional OBS-based
    selection.

    When no_obs=True, applies plain vanilla SVD truncation (no Fisher
    collection, no OBS correction). Useful as a baseline for comparison.

    Each weight matrix W is replaced by two smaller matrices:
        W ≈ (U √Σ) (√Σ Vᵀ)

    Args:
        model:          Model to compress.
        calib_data:     List of calibration batches (dicts with
                        "input_ids" and "attention_mask").
                        Not required when no_obs=True.
        ratio:          Compression ratio (fraction of parameters to keep).
        obs_scale:      Step-size for OBS correction delta.
        obs_damping:    Diagonal regularization for H_SS solve.
        obs_hdamping:   Diagonal regularization for selection scoring.
        alpha:          Candidate pool fraction beyond rank.
        select_by_loss: If True, use OBS sensitivity for selection.
        no_obs:         If True, skip OBS correction (plain SVD baseline).
        obs_batches:    Number of backward passes for Fisher estimation.
        device:         CUDA device string.
        hbar_save_path: Path to save/load precomputed Hbars.
        reuse_hbars:    If True and path exists, skip Fisher collection.
        target_layers:  Optional list of layer names to compress.
                        If None, all nn.Linear layers are compressed.

    Returns:
        The compressed model (modified in-place).
    """
    if not no_obs and calib_data is None:
        raise ValueError("calib_data is required when no_obs=False. "
                         "Provide calibration data or set no_obs=True.")

    model.eval()
    method_name = "plain SVD" if no_obs else "SVD-Surgeon"

    # ------------------------------------------------------------------
    # Step 1: Identify target layers and compute SVD info
    # ------------------------------------------------------------------
    all_linears = _find_linear_layers(model)
    if target_layers is not None:
        all_linears = {k: v for k, v in all_linears.items() if k in target_layers}

    print(f"Compressing {len(all_linears)} linear layers at ratio={ratio} "
          f"using {method_name}...")

    layer_svd_info = {}
    with torch.no_grad():
        for name, module in all_linears.items():
            W = module.weight.data.float().to(device)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            rank = _compute_rank(W.shape, ratio)
            full_k = rank + int((S.numel() - rank) * alpha)

            print(f"  {name}: shape {tuple(W.shape)}, "
                  f"rank {rank}/{full_k}/{S.numel()}")

            layer_svd_info[name] = {
                "U": U[:, :full_k].detach().cpu(),
                "S": S.detach().cpu(),
                "Vh": Vh[:full_k, :].detach().cpu(),
                "rank": rank,
                "full_k": full_k,
            }
            del W, U, S, Vh
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Collect spectral Fisher (skip for plain SVD)
    # ------------------------------------------------------------------
    Hbars = None
    if not no_obs:
        print("Collecting spectral Fisher...")
        Hbars = collect_spectral_fisher(
            model, calib_data, layer_svd_info,
            num_batches=obs_batches, device=device,
            hbar_save_path=hbar_save_path, reuse_hbars=reuse_hbars)

    # ------------------------------------------------------------------
    # Step 3: Apply SVD truncation (with or without OBS correction)
    # ------------------------------------------------------------------
    print(f"Applying {method_name}...")
    with torch.no_grad():
        for name, module in all_linears.items():
            info = layer_svd_info[name]
            S = info["S"].to(device)
            U = info["U"].to(device)
            Vh = info["Vh"].to(device)
            rank = info["rank"]
            dtype = module.weight.dtype

            if no_obs:
                # Plain SVD: keep top-rank singular values by magnitude
                sigma_new = S[:rank]
                keep_idx = torch.arange(rank, device=device)
            else:
                # SVD-Surgeon: OBS-corrected singular values
                Hbar = Hbars[name].to(device)
                sigma_new, keep_idx, _ = obs_correct_singular_values(
                    S=S, rank=rank, Hbar=Hbar,
                    obs_scale=obs_scale,
                    obs_damping=obs_damping,
                    obs_hdamping=obs_hdamping,
                    alpha=alpha,
                    select_by_loss=select_by_loss)
                del Hbar

            # Build low-rank factors
            truc_u = U[:, keep_idx]
            truc_v = Vh[keep_idx, :]
            truc_sigma = torch.diag(sigma_new)
            sqrt_sigma = torch.sqrt(truc_sigma)
            factor_u = torch.matmul(truc_u, sqrt_sigma).cpu().to(dtype)   # (out, rank)
            factor_v = torch.matmul(sqrt_sigma, truc_v).cpu().to(dtype)   # (rank, in)

            # Replace the original linear layer with two sequential linears
            out_dim, in_dim = module.weight.shape
            has_bias = module.bias is not None

            layer_v = nn.Linear(in_dim, rank, bias=False)
            layer_u = nn.Linear(rank, out_dim, bias=has_bias)

            layer_v.weight.data = factor_v
            layer_u.weight.data = factor_u
            if has_bias:
                layer_u.bias.data = module.bias.data.clone()

            # Replace in model
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], nn.Sequential(layer_v, layer_u))

            print(f"  {name}: {tuple(module.weight.shape)} -> "
                  f"({in_dim}, {rank}) + ({rank}, {out_dim})")

            del S, U, Vh, sigma_new, factor_u, factor_v
            torch.cuda.empty_cache()

    print(f"Compression complete ({method_name}).")
    return model
