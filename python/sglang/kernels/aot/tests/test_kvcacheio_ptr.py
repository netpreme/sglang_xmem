"""
Tests for the _ptr variants of kvcacheio transfer functions.

Each test runs both the tensor-based reference implementation and the
corresponding _ptr (raw integer pointer) variant on the same source data and
compares results.  The _ptr variants accept raw device pointers as Python ints
instead of torch.Tensor objects, which mirrors the usage pattern where callers
manage their own memory pools and only have raw addresses available.

Layout conventions (matching the C kernel naming):
  lf  - layer-first:  [layer_id, token_id, item]
  pf  - page-first:   flat buffer, offset = token_id * layout_dim + layer_id * item_bytes
  ph  - page-head:    [page_num, head_num, page_size, layer_num, head_dim]
"""

import pytest
import torch
from sgl_kernel.kvcacheio import (
    # _ptr variants under test
    transfer_kv_all_layer_lf_pf_ptr,
    transfer_kv_all_layer_lf_ph_ptr,
    transfer_kv_all_layer_mla_lf_pf_ptr,
    transfer_kv_all_layer_mla_ptr,
    transfer_kv_all_layer_ptr,
    transfer_kv_per_layer_mla_pf_lf_ptr,
    transfer_kv_per_layer_mla_ptr,
    transfer_kv_per_layer_pf_lf_ptr,
    transfer_kv_per_layer_ph_lf_ptr,
    transfer_kv_per_layer_ptr,
    # tensor-based references
    transfer_kv_all_layer,
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_all_layer_mla_lf_pf,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_mla_pf_lf,
    transfer_kv_per_layer_pf_lf,
    transfer_kv_per_layer_ph_lf,
)

from sglang.srt.utils import is_hip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indices(total_items_in_pool, num_items_to_transfer, page_size, device):
    """Return (src_indices, dst_indices) as CUDA int64 tensors."""
    total_pages = total_items_in_pool // page_size
    num_pages = num_items_to_transfer // page_size
    page_idx = torch.randperm(total_pages, dtype=torch.int64)
    src = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_idx[:num_pages]
        ]
    ).to(device)
    dst = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_idx[num_pages : 2 * num_pages]
        ]
    ).to(device)
    return src, dst


def _layer_ptr_table(layer_tensors, device):
    """Build a uint64 CUDA tensor of data_ptr() values for each layer tensor."""
    return torch.tensor(
        [t.data_ptr() for t in layer_tensors], dtype=torch.uint64, device=device
    )


# ---------------------------------------------------------------------------
# Test 1: lf <-> lf  (_per_layer_ptr, _all_layer_ptr, MLA variants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [1, 128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv_ptr(
    dtype,
    num_items_to_transfer,
    item_size,
    page_size,
    total_items_in_pool,
    is_mla,
    all_layers,
):
    """
    _ptr variants for lf<->lf transfers produce identical output to tensor variants.

    Covers:
      transfer_kv_per_layer_ptr        (all_layers=False, is_mla=False)
      transfer_kv_per_layer_mla_ptr    (all_layers=False, is_mla=True)
      transfer_kv_all_layer_ptr        (all_layers=True,  is_mla=False)
      transfer_kv_all_layer_mla_ptr    (all_layers=True,  is_mla=True)
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    src_indices, dst_indices = _make_indices(
        total_items_in_pool, num_items_to_transfer, page_size, device
    )
    item_size_bytes = item_size * dtype.itemsize
    layer_idx = 0

    if is_mla:
        # MLA: single kv tensor per layer (no separate v).
        src_pool = [
            torch.randn(total_items_in_pool, item_size, device=device)
            for _ in range(num_layers)
        ]

        if not all_layers:
            dst_ref = torch.zeros(total_items_in_pool, item_size, device=device)
            dst_ptr = torch.zeros_like(dst_ref)

            transfer_kv_per_layer_mla(
                src_pool[layer_idx], dst_ref, src_indices, dst_indices,
                item_size=item_size_bytes,
            )
            transfer_kv_per_layer_mla_ptr(
                src_pool[layer_idx].data_ptr(),
                dst_ptr.data_ptr(),
                src_indices,
                dst_indices,
                item_size=item_size_bytes,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_ptr, dst_ref)
        else:
            dst_ref = [
                torch.zeros(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]
            dst_ptr_layers = [
                torch.zeros(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]

            src_tbl = _layer_ptr_table(src_pool, device)
            dst_ref_tbl = _layer_ptr_table(dst_ref, device)
            dst_ptr_tbl = _layer_ptr_table(dst_ptr_layers, device)

            transfer_kv_all_layer_mla(
                src_tbl, dst_ref_tbl, src_indices, dst_indices,
                item_size=item_size_bytes, num_layers=num_layers,
            )
            transfer_kv_all_layer_mla_ptr(
                src_tbl.data_ptr(),
                dst_ptr_tbl.data_ptr(),
                src_indices,
                dst_indices,
                item_size=item_size_bytes,
                num_layers=num_layers,
            )
            torch.cuda.synchronize()
            for i in range(num_layers):
                torch.testing.assert_close(dst_ptr_layers[i], dst_ref[i])
    else:
        # Standard KV: separate k and v tensors per layer.
        src_k_pool = [
            torch.randn(total_items_in_pool, item_size, device=device)
            for _ in range(num_layers)
        ]
        src_v_pool = [
            torch.randn(total_items_in_pool, item_size, device=device)
            for _ in range(num_layers)
        ]

        if not all_layers:
            dst_k_ref = torch.zeros(total_items_in_pool, item_size, device=device)
            dst_v_ref = torch.zeros_like(dst_k_ref)
            dst_k_ptr = torch.zeros_like(dst_k_ref)
            dst_v_ptr = torch.zeros_like(dst_v_ref)

            transfer_kv_per_layer(
                src_k_pool[layer_idx], dst_k_ref,
                src_v_pool[layer_idx], dst_v_ref,
                src_indices, dst_indices,
                item_size=item_size_bytes,
            )
            transfer_kv_per_layer_ptr(
                src_k_pool[layer_idx].data_ptr(), dst_k_ptr.data_ptr(),
                src_v_pool[layer_idx].data_ptr(), dst_v_ptr.data_ptr(),
                src_indices, dst_indices,
                item_size=item_size_bytes,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_ptr, dst_k_ref)
            torch.testing.assert_close(dst_v_ptr, dst_v_ref)
        else:
            dst_k_ref = [
                torch.zeros(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]
            dst_v_ref = [torch.zeros_like(dst_k_ref[0]) for _ in range(num_layers)]
            dst_k_ptr_layers = [
                torch.zeros_like(dst_k_ref[0]) for _ in range(num_layers)
            ]
            dst_v_ptr_layers = [
                torch.zeros_like(dst_v_ref[0]) for _ in range(num_layers)
            ]

            src_k_tbl = _layer_ptr_table(src_k_pool, device)
            src_v_tbl = _layer_ptr_table(src_v_pool, device)
            dst_k_ref_tbl = _layer_ptr_table(dst_k_ref, device)
            dst_v_ref_tbl = _layer_ptr_table(dst_v_ref, device)
            dst_k_ptr_tbl = _layer_ptr_table(dst_k_ptr_layers, device)
            dst_v_ptr_tbl = _layer_ptr_table(dst_v_ptr_layers, device)

            transfer_kv_all_layer(
                src_k_tbl, dst_k_ref_tbl, src_v_tbl, dst_v_ref_tbl,
                src_indices, dst_indices,
                item_size=item_size_bytes, num_layers=num_layers,
            )
            transfer_kv_all_layer_ptr(
                src_k_tbl.data_ptr(), dst_k_ptr_tbl.data_ptr(),
                src_v_tbl.data_ptr(), dst_v_ptr_tbl.data_ptr(),
                src_indices, dst_indices,
                item_size=item_size_bytes, num_layers=num_layers,
            )
            torch.cuda.synchronize()
            for i in range(num_layers):
                torch.testing.assert_close(dst_k_ptr_layers[i], dst_k_ref[i])
                torch.testing.assert_close(dst_v_ptr_layers[i], dst_v_ref[i])

    torch.set_default_dtype(original_dtype)


# ---------------------------------------------------------------------------
# Test 2: pf <-> lf  (_per_layer_pf_lf_ptr, _all_layer_lf_pf_ptr, MLA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_lf_ptr(
    dtype,
    num_items_to_transfer,
    item_size,
    page_size,
    total_items_in_pool,
    is_mla,
    lf_to_pf,
):
    """
    _ptr variants for pf<->lf transfers produce identical output to tensor variants.

    The pf (page-first) layout stores all layers for a token contiguously:
      byte offset = token_id * layout_dim + layer_id * item_size_bytes
    where layout_dim = num_layers * item_size_bytes.

    Covers:
      transfer_kv_per_layer_pf_lf_ptr      (lf_to_pf=False, is_mla=False)
      transfer_kv_per_layer_mla_pf_lf_ptr  (lf_to_pf=False, is_mla=True)
      transfer_kv_all_layer_lf_pf_ptr      (lf_to_pf=True,  is_mla=False)
      transfer_kv_all_layer_mla_lf_pf_ptr  (lf_to_pf=True,  is_mla=True)
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    src_indices, dst_indices = _make_indices(
        total_items_in_pool, num_items_to_transfer, page_size, device
    )
    item_size_bytes = item_size * dtype.itemsize
    # pf layout_dim: total bytes per token across all layers
    layout_dim = num_layers * item_size_bytes
    layer_idx = 0

    if lf_to_pf:
        # src: lf (per-layer pointer table)  ->  dst: pf flat tensor
        # dst shape: (total_items_in_pool, num_layers * item_size) in dtype elements
        if is_mla:
            src_pool = [
                torch.randn(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]
            src_tbl = _layer_ptr_table(src_pool, device)

            dst_ref = torch.zeros(
                total_items_in_pool, num_layers * item_size, device=device
            )
            dst_ptr = torch.zeros_like(dst_ref)

            transfer_kv_all_layer_mla_lf_pf(
                src_tbl, dst_ref, src_indices, dst_indices,
                item_size=item_size_bytes,
                dst_layout_dim=layout_dim,
                num_layers=num_layers,
            )
            transfer_kv_all_layer_mla_lf_pf_ptr(
                src_tbl.data_ptr(),
                dst_ptr.data_ptr(),
                src_indices,
                dst_indices,
                item_size=item_size_bytes,
                dst_layout_dim=layout_dim,
                num_layers=num_layers,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_ptr, dst_ref)
        else:
            src_k_pool = [
                torch.randn(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]
            src_v_pool = [
                torch.randn(total_items_in_pool, item_size, device=device)
                for _ in range(num_layers)
            ]
            src_k_tbl = _layer_ptr_table(src_k_pool, device)
            src_v_tbl = _layer_ptr_table(src_v_pool, device)

            dst_k_ref = torch.zeros(
                total_items_in_pool, num_layers * item_size, device=device
            )
            dst_v_ref = torch.zeros_like(dst_k_ref)
            dst_k_ptr = torch.zeros_like(dst_k_ref)
            dst_v_ptr = torch.zeros_like(dst_v_ref)

            transfer_kv_all_layer_lf_pf(
                src_k_tbl, dst_k_ref, src_v_tbl, dst_v_ref,
                src_indices, dst_indices,
                item_size=item_size_bytes,
                dst_layout_dim=layout_dim,
                num_layers=num_layers,
            )
            transfer_kv_all_layer_lf_pf_ptr(
                src_k_tbl.data_ptr(), dst_k_ptr.data_ptr(),
                src_v_tbl.data_ptr(), dst_v_ptr.data_ptr(),
                src_indices, dst_indices,
                item_size=item_size_bytes,
                dst_layout_dim=layout_dim,
                num_layers=num_layers,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_ptr, dst_k_ref)
            torch.testing.assert_close(dst_v_ptr, dst_v_ref)
    else:
        # src: pf flat tensor  ->  dst: lf single-layer tensor
        # src shape: (total_items_in_pool, num_layers * item_size) in dtype elements
        if is_mla:
            src_pf = torch.randn(
                total_items_in_pool, num_layers * item_size, device=device
            )
            dst_ref = torch.zeros(total_items_in_pool, item_size, device=device)
            dst_ptr = torch.zeros_like(dst_ref)

            transfer_kv_per_layer_mla_pf_lf(
                src_pf, dst_ref, src_indices, dst_indices,
                layer_id=layer_idx,
                item_size=item_size_bytes,
                src_layout_dim=layout_dim,
            )
            transfer_kv_per_layer_mla_pf_lf_ptr(
                src_pf.data_ptr(),
                dst_ptr.data_ptr(),
                src_indices,
                dst_indices,
                layer_id=layer_idx,
                item_size=item_size_bytes,
                src_layout_dim=layout_dim,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_ptr, dst_ref)
        else:
            src_k_pf = torch.randn(
                total_items_in_pool, num_layers * item_size, device=device
            )
            src_v_pf = torch.randn(
                total_items_in_pool, num_layers * item_size, device=device
            )
            dst_k_ref = torch.zeros(total_items_in_pool, item_size, device=device)
            dst_v_ref = torch.zeros_like(dst_k_ref)
            dst_k_ptr = torch.zeros_like(dst_k_ref)
            dst_v_ptr = torch.zeros_like(dst_v_ref)

            transfer_kv_per_layer_pf_lf(
                src_k_pf, dst_k_ref, src_v_pf, dst_v_ref,
                src_indices, dst_indices,
                layer_id=layer_idx,
                item_size=item_size_bytes,
                src_layout_dim=layout_dim,
            )
            transfer_kv_per_layer_pf_lf_ptr(
                src_k_pf.data_ptr(), dst_k_ptr.data_ptr(),
                src_v_pf.data_ptr(), dst_v_ptr.data_ptr(),
                src_indices, dst_indices,
                layer_id=layer_idx,
                item_size=item_size_bytes,
                src_layout_dim=layout_dim,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_ptr, dst_k_ref)
            torch.testing.assert_close(dst_v_ptr, dst_v_ref)

    torch.set_default_dtype(original_dtype)


# ---------------------------------------------------------------------------
# Test 3: page-head <-> lf  (_per_layer_ph_lf_ptr, _all_layer_lf_ph_ptr)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [1024])
@pytest.mark.parametrize("head_num", [8, 16])
@pytest.mark.parametrize("total_items_in_pool", [4096])
@pytest.mark.parametrize("lf_to_ph", [False, True])
def test_transfer_kv_page_head_ptr(
    dtype,
    num_items_to_transfer,
    page_size,
    item_size,
    head_num,
    total_items_in_pool,
    lf_to_ph,
):
    """
    _ptr variants for page-head <-> lf transfers produce identical output to tensor variants.

    The ph (page-head) layout is: [page_num, head_num, page_size, layer_num, head_dim]

    Covers:
      transfer_kv_all_layer_lf_ph_ptr   (lf_to_ph=True)
      transfer_kv_per_layer_ph_lf_ptr   (lf_to_ph=False)
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4
    assert item_size % head_num == 0
    head_dim = item_size // head_num
    total_pages = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    src_indices, dst_indices = _make_indices(
        total_items_in_pool, num_items_to_transfer, page_size, device
    )
    item_size_bytes = item_size * dtype.itemsize
    # layout_dim for ph: bytes per token across all layers and heads
    # = item_size_bytes * num_layers  (since item_size = head_num * head_dim)
    layout_dim = item_size_bytes * num_layers
    layer_idx = 0

    if lf_to_ph:
        # src: lf (per-layer tensors, shape (total_items_in_pool, head_num, head_dim))
        # dst: ph tensor, shape (total_pages, head_num, page_size, num_layers, head_dim)
        src_k_pool = [
            torch.randn(total_items_in_pool, head_num, head_dim, device=device)
            for _ in range(num_layers)
        ]
        src_v_pool = [
            torch.randn(total_items_in_pool, head_num, head_dim, device=device)
            for _ in range(num_layers)
        ]
        src_k_tbl = _layer_ptr_table(src_k_pool, device)
        src_v_tbl = _layer_ptr_table(src_v_pool, device)

        dst_k_ref = torch.zeros(
            total_pages, head_num, page_size, num_layers, head_dim, device=device
        )
        dst_v_ref = torch.zeros_like(dst_k_ref)
        dst_k_ptr = torch.zeros_like(dst_k_ref)
        dst_v_ptr = torch.zeros_like(dst_v_ref)

        transfer_kv_all_layer_lf_ph(
            src_k_tbl, dst_k_ref, src_v_tbl, dst_v_ref,
            src_indices, dst_indices,
            item_size_bytes, layout_dim, num_layers, page_size, head_num,
        )
        transfer_kv_all_layer_lf_ph_ptr(
            src_k_tbl.data_ptr(), dst_k_ptr.data_ptr(),
            src_v_tbl.data_ptr(), dst_v_ptr.data_ptr(),
            src_indices, dst_indices,
            item_size_bytes, layout_dim, num_layers, page_size, head_num,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_ptr, dst_k_ref)
        torch.testing.assert_close(dst_v_ptr, dst_v_ref)
    else:
        # src: ph tensor, shape (total_pages, head_num, page_size, num_layers, head_dim)
        # dst: lf single-layer tensor, shape (total_items_in_pool, head_num, head_dim)
        src_k_pool = torch.randn(
            total_pages, head_num, page_size, num_layers, head_dim, device=device
        )
        src_v_pool = torch.randn(
            total_pages, head_num, page_size, num_layers, head_dim, device=device
        )

        dst_k_ref = torch.zeros(
            total_items_in_pool, head_num, head_dim, device=device
        )
        dst_v_ref = torch.zeros_like(dst_k_ref)
        dst_k_ptr = torch.zeros_like(dst_k_ref)
        dst_v_ptr = torch.zeros_like(dst_v_ref)

        transfer_kv_per_layer_ph_lf(
            src_k_pool, dst_k_ref, src_v_pool, dst_v_ref,
            src_indices, dst_indices,
            layer_idx, item_size_bytes, layout_dim, page_size, head_num,
        )
        transfer_kv_per_layer_ph_lf_ptr(
            src_k_pool.data_ptr(), dst_k_ptr.data_ptr(),
            src_v_pool.data_ptr(), dst_v_ptr.data_ptr(),
            src_indices, dst_indices,
            layer_idx, item_size_bytes, layout_dim, page_size, head_num,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_ptr, dst_k_ref)
        torch.testing.assert_close(dst_v_ptr, dst_v_ref)

    torch.set_default_dtype(original_dtype)


if __name__ == "__main__":
    pytest.main([__file__])
