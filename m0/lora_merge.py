"""Chirurgie d'adaptateurs LoRA (M3) : empilage (concat) + rebase SVD.

Convention mlx_lm (verifiee) : par module, lora_a (in, r), lora_b (r, out), et le
delta effectif applique est  scale * (x @ a) @ b  =  x @ (scale * a@b).
Le "delta low-rank" d'un module est donc  M = a @ b  (in, out), applique avec `scale`.

- concat(ad1, ad2) : a=[a1|a2] (in,2r), b=[b1;b2] (2r,out) -> SOMME des deux deltas,
  representee au rang 2r (le rang s'ADDITIONNE). Scale inchange.
- svd_rebase(ad, r) : tronque chaque module au rang r (meilleure approx, Eckart-Young).

IMPORTANT perf : M = a@b est de rang <= R (petit). On NE forme jamais la grande matrice
M (in×out peut etre 18944×3584). On calcule la SVD via les facteurs (QR sur tall-skinny
+ SVD R×R), c'est exact et quasi-instantane. (numpy/CPU : la SVD Metal/mlx timeoute.)
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np


def load_adapter(path: str) -> dict:
    w = mx.load(os.path.join(path, "adapters.safetensors"))
    return {k: np.array(v, dtype=np.float32) for k, v in w.items()}


def _modules(ad: dict) -> list[str]:
    return sorted({k[:-7] for k in ad if k.endswith(".lora_a")})


def concat_adapters(ad1: dict, ad2: dict) -> dict:
    """Empile deux adaptateurs de meme structure -> rang additionne (somme des deltas)."""
    out: dict = {}
    for base in _modules(ad1):
        out[base + ".lora_a"] = np.concatenate(
            [ad1[base + ".lora_a"], ad2[base + ".lora_a"]], axis=1)
        out[base + ".lora_b"] = np.concatenate(
            [ad1[base + ".lora_b"], ad2[base + ".lora_b"]], axis=0)
    return out


def _lowrank_svd(a: np.ndarray, b: np.ndarray):
    """SVD de M = a@b SANS former M. a:(in,R), b:(R,out). Retourne U(in,R), S(R), Vt(R,out)."""
    Qa, Ra = np.linalg.qr(a)        # (in,R), (R,R)
    Qb, Rb = np.linalg.qr(b.T)      # (out,R), (R,R)
    X = Ra @ Rb.T                   # (R,R)
    Ux, S, Vxt = np.linalg.svd(X)   # (R,R)
    U = Qa @ Ux                     # (in,R)
    Vt = Vxt @ Qb.T                 # (R,out)
    return U, S, Vt


def svd_rebase(ad: dict, target_rank: int) -> dict:
    """Tronque chaque module au rang `target_rank` via SVD du delta (Eckart-Young)."""
    out: dict = {}
    for base in _modules(ad):
        U, S, Vt = _lowrank_svd(ad[base + ".lora_a"], ad[base + ".lora_b"])
        r = min(target_rank, S.shape[0])
        sq = np.sqrt(S[:r])
        out[base + ".lora_a"] = (U[:, :r] * sq[None, :]).astype(np.float32)
        out[base + ".lora_b"] = (sq[:, None] * Vt[:r, :]).astype(np.float32)
    return out


def truncation_energy(ad: dict, target_rank: int) -> float:
    """Fraction d'energie spectrale (somme des sigma^2) conservee a un rang donne, agregee.
    1.0 = compression sans perte au sens Frobenius (Eckart-Young)."""
    kept = tot = 0.0
    for base in _modules(ad):
        _, S, _ = _lowrank_svd(ad[base + ".lora_a"], ad[base + ".lora_b"])
        r = min(target_rank, S.shape[0])
        kept += float((S[:r] ** 2).sum())
        tot += float((S ** 2).sum())
    return kept / max(tot, 1e-9)


def _randomized_svd(M: np.ndarray, r: int, oversample: int = 8, n_iter: int = 2):
    """SVD tronquee approchee (top-r) d'une matrice PLEIN-rang, sans full SVD.
    Necessaire apres TIES (le merge element-wise casse la structure low-rank)."""
    rng = np.random.default_rng(0)
    m, n = M.shape
    p = min(r + oversample, m, n)
    Y = M @ rng.standard_normal((n, p)).astype(np.float32)
    for _ in range(n_iter):  # power iterations (precision)
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)
    Ub, S, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)
    U = Q @ Ub
    r = min(r, S.shape[0])
    return U[:, :r], S[:r], Vt[:r, :]


def _trim(M: np.ndarray, density: float) -> np.ndarray:
    """Garde la fraction `density` des entrees de plus forte magnitude, zero le reste."""
    if density >= 1.0:
        return M
    k = max(1, int(density * M.size))
    thresh = np.partition(np.abs(M).ravel(), M.size - k)[M.size - k]
    return np.where(np.abs(M) >= thresh, M, np.float32(0.0))


def ties_merge(*adapters, density: float = 0.2, target_rank: int = 16) -> dict:
    """Merge TIES de N LoRA (Trim, Elect-sign, Merge) sur les deltas M=a@b, puis
    refactorisation rang-r. Resout l'interference du concat naif : on trim, on elit le
    signe dominant par entree, on moyenne seulement les contributions du bon signe.
    Appel : ties_merge(adA, adB[, adC...], density=..., target_rank=...)."""
    import gc

    if len(adapters) == 1 and isinstance(adapters[0], (list, tuple)):
        adapters = tuple(adapters[0])
    if len(adapters) < 2:
        raise ValueError("ties_merge attend >=2 adaptateurs")

    # union des modules : les adaptateurs peuvent differer (num_layers varie via la gate)
    all_bases = sorted(set().union(*[set(_modules(ad)) for ad in adapters]))
    out: dict = {}
    for base in all_bases:
        present = [ad for ad in adapters if (base + ".lora_a") in ad]
        Ts = [_trim((ad[base + ".lora_a"] @ ad[base + ".lora_b"]).astype(np.float32), density)
              for ad in present]
        elected = np.sign(sum(Ts))
        num = np.zeros_like(Ts[0])
        cnt = np.zeros_like(Ts[0])
        for T in Ts:
            keep = (np.sign(T) == elected) & (elected != 0)
            num += np.where(keep, T, 0.0)
            cnt += keep.astype(np.float32)
        M = np.where(cnt > 0, num / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
        U, S, Vt = _randomized_svd(M, target_rank)
        sq = np.sqrt(S)
        out[base + ".lora_a"] = (U * sq[None, :]).astype(np.float32)
        out[base + ".lora_b"] = (sq[:, None] * Vt).astype(np.float32)
        del Ts, elected, num, cnt, M
        gc.collect()
    return out


def save_adapter(ad: dict, out_dir: str, rank: int, src_config_path: str) -> None:
    """Ecrit adapters.safetensors + adapter_config.json (rang mis a jour) pour mlx_lm."""
    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(
        os.path.join(out_dir, "adapters.safetensors"),
        {k: mx.array(v) for k, v in ad.items()},
    )
    with open(src_config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("lora_parameters", {})["rank"] = int(rank)
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
