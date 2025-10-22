import take_along_dim_grouped
import torch
import triton
import triton.language as tl


def d0_index_select(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    src: (-1, ndim); in cpu
    indices: (n, ); in cpu
    """
    out = torch.empty(
        indices.shape[0], src.shape[1],
        dtype=src.dtype, device=src.device, pin_memory=True)
    take_along_dim_grouped.d0_indices_select(out, src, indices)
    return out

def take_along_dim_with_mask_python_linear_indices(
    seq_len,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    # assert src.dim() == 4, "src must be 4D [bsz, kv_heads, seq_len, hdim]"
    # assert indices.dim(
    # ) == 4, "indices must be 4D [bsz, kv_heads, num_groups, k]"
    # assert indices_mask.shape == indices.shape, "mask shape must match indices"
    # assert indices_mask.dtype == torch.bool, "mask must be bool tensor"

    # Get the coordinates of the non-zero elements in the mask
    coords = indices_mask.nonzero(as_tuple=False)  # [num_valid,4] on GPU
    # Get the batch size and key-value heads
    i = coords[:, 0]
    j = coords[:, 1]
    # Get the indices of the valid elements
    idx_val = indices[indices_mask]

    kv_heads = indices.shape[1]

    # Calculate the linear indices
    linear_indices = (i * kv_heads * seq_len) + \
        (j * seq_len) + idx_val  # [num_valid]
    
    return linear_indices

def take_along_dim_with_mask_naive(
    seq_len: int,
    indices: torch.Tensor,
    indices_mask: torch.Tensor,
) -> torch.Tensor:
    assert indices_mask.shape == indices.shape

    # Ensure contiguous tensors for correct memory access
    indices_mask = indices_mask.contiguous()
    indices = indices.contiguous()

    # Get tensor dimensions
    bsz, kv_heads, num_groups, k = indices_mask.shape

    # Precompute number of valid elements for output allocation
    num_valid = indices_mask.sum().item()
    linear_indices = torch.empty(
        num_valid, dtype=torch.int64, device=indices.device)

    pos = 0
    for _b in range(bsz):
        for _kv in range(kv_heads):
            for _g in range(num_groups):
                for _k in range(k):
                    if indices_mask[_b, _kv, _g, _k]:
                        linear_indices[pos] = _b * kv_heads * seq_len + _kv * seq_len + indices[_b, _kv, _g, _k]
                        pos += 1


    return linear_indices


@triton.jit
def take_along_dim_kernel_v0(
    indices_flatten_ptr,
    mask_flatten_ptr,
    prefix_ptr,
    linear_indices_ptr,
    total_elements,
    kv_heads,
    num_groups,
    k,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Load mask values and check validity
    mask_vals = tl.load(mask_flatten_ptr + offs, mask=mask, other=0)
    is_valid = mask_vals != 0

    # Calculate 4D indices (b, kv, g, k_idx) from flattened index
    elements_per_b = kv_heads * num_groups * k
    b = offs // elements_per_b
    i_remain = offs % elements_per_b

    elements_per_kv = num_groups * k
    kv = i_remain // elements_per_kv
    # i_remain = i_remain % elements_per_kv

    # Load the index value
    indices_val = tl.load(indices_flatten_ptr + offs, mask=mask, other=0)

    # Compute linear index
    linear_index = b * kv_heads * seq_len + kv * seq_len + indices_val

    # Get position in output array
    pos = tl.load(prefix_ptr + offs, mask=mask, other=0)

    # Store result if valid
    tl.store(linear_indices_ptr + pos, linear_index, mask=mask & is_valid)

def take_along_dim_with_mask_triton_v0(
    seq_len: int,
    indices: torch.Tensor,
    indices_mask: torch.Tensor,
) -> torch.Tensor:
    assert indices_mask.shape == indices.shape

    _, kv_heads, num_groups, k = indices_mask.shape

    # Flatten tensors
    mask_flatten = indices_mask.view(-1)
    indices_flatten = indices.view(-1)

    mask_flatten_int32 = mask_flatten.to(torch.int32)

    # Compute prefix sum for output positions
    prefix = (torch.cumsum(mask_flatten_int32, dim=0) - mask_flatten_int32)
    num_valid = prefix[-1].item() + mask_flatten_int32[-1].item()

    # Allocate output
    linear_indices = torch.empty(num_valid, dtype=torch.int64, device=indices.device)

    total_elements = mask_flatten_int32.numel()
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(total_elements, BLOCK_SIZE), )

    # Launch kernel
    take_along_dim_kernel[grid](
        indices_flatten,
        mask_flatten_int32,
        prefix,
        linear_indices,
        total_elements,
        kv_heads,
        num_groups,
        k,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return linear_indices


@triton.jit
def take_along_dim_kernel(
    indices_flatten_ptr,
    mask_flatten_ptr,
    prefix_ptr,
    linear_indices_ptr,
    total_elements,
    kv_heads,
    num_groups,
    k,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Precompute constants once per kernel
    elements_per_b = kv_heads * num_groups * k
    elements_per_kv = num_groups * k

    # Calculate 4D indices (b, kv, g, k_idx) from flattened index
    b = offs // elements_per_b
    i_remain = offs % elements_per_b
    kv = i_remain // elements_per_kv

    # Load mask values and check validity
    mask_vals = tl.load(mask_flatten_ptr + offs, mask=mask, other=0)
    is_valid = mask_vals != 0

    # Load index values and compute linear index
    indices_val = tl.load(indices_flatten_ptr + offs, mask=mask, other=0)
    linear_index = (b * kv_heads + kv) * seq_len + indices_val

    # Get position in output array
    pos = tl.load(prefix_ptr + offs, mask=mask, other=0)

    # Store result if valid
    tl.store(linear_indices_ptr + pos, linear_index, mask=mask & is_valid)


def take_along_dim_with_mask_triton(
    seq_len: int,
    indices: torch.Tensor,
    indices_mask: torch.Tensor,
) -> torch.Tensor:
    assert indices_mask.shape == indices.shape

    _, kv_heads, num_groups, k = indices_mask.shape

    # Flatten tensors
    mask_flatten = indices_mask.reshape(-1)
    indices_flatten = indices.reshape(-1)

    mask_flatten_int32 = mask_flatten.to(torch.int32)

    # Compute prefix sum for output positions
    prefix = (torch.cumsum(mask_flatten_int32, dim=0) - mask_flatten_int32)
    num_valid = prefix[-1].item() + mask_flatten_int32[-1].item()

    # Allocate output
    linear_indices = torch.empty(
        num_valid, dtype=torch.int64, device=indices.device)

    total_elements = mask_flatten_int32.numel()
    max_block_size = 4096

    # Calculate optimal block size
    block_size = min(
        triton.next_power_of_2(total_elements),
        max_block_size
    )
    block_size = max(block_size, 16)  # Minimum of 16 threads

    grid = (triton.cdiv(total_elements, block_size), )

    # BLOCK_SIZE = 512  # Adjusted for better occupancy
    # grid = (triton.cdiv(total_elements, BLOCK_SIZE), )

    # Launch kernel
    take_along_dim_kernel[grid](
        indices_flatten,
        mask_flatten_int32,
        prefix,
        linear_indices,
        total_elements,
        kv_heads,
        num_groups,
        k,
        seq_len,
        BLOCK_SIZE=block_size,
    )

    return linear_indices


