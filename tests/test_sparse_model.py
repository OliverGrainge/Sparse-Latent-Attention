"""Sparse latent attention wired into the full model: cached decoding (torch
reference and Triton kernels) must agree with a single full forward pass.

Top-k selection is discontinuous, so with a random indexer tiny numerical
differences between prefill and decode can flip which keys are kept and then
cascade through the layers. The selection test therefore runs in fp32; the
Triton test uses k >= context so every key is kept and the kernel itself is
what is being compared.
"""

from pathlib import Path

import pytest
import torch

from sla.model import load_qwen3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
LATENT = Path("checkpoints/latent_r512/latent.pt")


def _model(dtype):
    if not LATENT.exists():
        pytest.skip("needs a trained latent checkpoint")
    m = load_qwen3("Qwen/Qwen3-1.7B", dtype=dtype)
    m.load_latent(LATENT)
    torch.manual_seed(0)
    m.add_indexer()
    with torch.no_grad():
        for layer in m.layers:
            layer.attn.indexer.ww.normal_()
    return m


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 150_000, (1, 160), generator=g).cuda()


def _decode_vs_full(model, ids, impl, n_pre=100):
    model.attn_impl = "sdpa"
    full = model(ids).float()
    model.attn_impl = impl
    cache = model.make_cache(1, ids.shape[1])
    steps = [model(ids[:, :n_pre], 0, cache).float()[:, -1]]
    for t in range(n_pre, ids.shape[1] - 1):
        steps.append(model(ids[:, t : t + 1], t, cache).float()[:, -1])
    model.attn_impl = "sdpa"
    dec, ref = torch.stack(steps, 1), full[:, n_pre - 1 : ids.shape[1] - 1]
    agree = (dec.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return agree, ((dec - ref).norm() / ref.norm()).item()


@torch.inference_mode()
def test_sparse_cached_decode_fp32(ids):
    model = _model(torch.float32)
    model.set_sparse(32)
    agree, rel = _decode_vs_full(model, ids, "sdpa")
    del model
    torch.cuda.empty_cache()
    assert agree == 1.0 and rel < 1e-3, (agree, rel)


@torch.inference_mode()
def test_triton_decode_bf16(ids):
    pytest.importorskip("triton")
    model = _model(torch.bfloat16)
    model.set_sparse(ids.shape[1])
    agree, rel = _decode_vs_full(model, ids, "triton")
    assert agree > 0.95 and rel < 0.03, (agree, rel)
