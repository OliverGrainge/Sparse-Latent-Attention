"""Checks for the MLA-style latent KV path."""

import pytest
import torch

from sla.latent import collect_kv_second_moments, init_latent_from_pca
from sla.model import KVCache, load_qwen3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def model():
    return load_qwen3("Qwen/Qwen3-1.7B", dtype=torch.float32)


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 150_000, (2, 128), generator=g).cuda()


@pytest.fixture(scope="module")
def moments(model, ids):
    return collect_kv_second_moments(model, [ids], progress=False)


@torch.inference_mode()
def test_full_rank_latent_is_exact(model, ids, moments):
    """With r = full KV dim, PCA keeps everything: latent model == dense model."""
    model.set_latent(False)
    dense = model(ids)
    kv_dim = 2 * model.cfg.num_kv_heads * model.cfg.head_dim
    stats = init_latent_from_pca(model, moments, rank=kv_dim)
    model.set_latent(True)
    try:
        latent = model(ids)
    finally:
        model.set_latent(False)
    assert max(s["k_rel_err"] for s in stats) < 1e-6
    assert (latent - dense).abs().max().item() < 5e-3


@torch.inference_mode()
def test_latent_cache_matches_full_forward(model, ids, moments):
    init_latent_from_pca(model, moments, rank=256)
    model.set_latent(True)
    try:
        full = model(ids)
        cache = KVCache(model.cfg, ids.shape[0], ids.shape[1], ids.device, torch.float32, latent_rank=256)
        a = model(ids[:, :100], start_pos=0, cache=cache)
        steps = [model(ids[:, t:t + 1], start_pos=t, cache=cache) for t in range(100, ids.shape[1])]
        assert (torch.cat([a, *steps], 1) - full).abs().max().item() < 2e-3
        assert cache.nbytes() == 28 * 2 * 128 * 256 * 4
    finally:
        model.set_latent(False)
