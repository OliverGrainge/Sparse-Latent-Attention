"""Correctness checks for the native-torch Qwen3 against the HF reference."""

import pytest
import torch

from sla.model import KVCache, load_qwen3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REPO = "Qwen/Qwen3-1.7B"


@pytest.fixture(scope="module")
def ours_fp32():
    return load_qwen3(REPO, dtype=torch.float32)


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 150_000, (2, 96), generator=g).cuda()


@torch.inference_mode()
def test_matches_hf_fp32(ours_fp32, ids):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(ours_fp32.path), dtype=torch.float32, attn_implementation="eager"
    ).cuda().eval()
    ref = hf(ids).logits
    del hf
    torch.cuda.empty_cache()

    out = ours_fp32(ids)
    err = (out - ref).abs().max().item()
    assert err < 2e-3, f"max |logit diff| = {err}"
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.999


@torch.inference_mode()
def test_sdpa_matches_eager(ours_fp32, ids):
    ours_fp32.attn_impl = "eager"
    a = ours_fp32(ids)
    ours_fp32.attn_impl = "sdpa"
    b = ours_fp32(ids)
    ours_fp32.attn_impl = "eager"
    assert (a - b).abs().max().item() < 2e-3


@torch.inference_mode()
def test_kv_cache_matches_full_forward(ours_fp32, ids):
    full = ours_fp32(ids)
    cache = KVCache(ours_fp32.cfg, ids.shape[0], ids.shape[1], ids.device, torch.float32)
    split = 64
    a = ours_fp32(ids[:, :split], start_pos=0, cache=cache)
    steps = [ours_fp32(ids[:, t : t + 1], start_pos=t, cache=cache) for t in range(split, ids.shape[1])]
    inc = torch.cat([a, *steps], dim=1)
    assert (inc - full).abs().max().item() < 2e-3


@torch.inference_mode()
def test_eager_chunking_is_exact(ours_fp32, ids):
    """Forcing tiny query chunks must not change the result."""
    import sla.model as m

    orig = m.ATTN_SCORE_BUDGET_BYTES
    full = ours_fp32(ids)
    m.ATTN_SCORE_BUDGET_BYTES = 1  # forces one query row per chunk
    try:
        chunked = ours_fp32(ids)
    finally:
        m.ATTN_SCORE_BUDGET_BYTES = orig
    assert (chunked - full).abs().max().item() < 1e-4
