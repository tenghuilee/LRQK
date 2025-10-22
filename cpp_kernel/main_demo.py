import unittest
import itertools

import numpy as np
import torch

# import cpp_kernel.take_along_dim_grouped as take_along_dim_grouped

import torch


def take_along_dim_with_mask_python_v0(
    src: torch.Tensor,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    # 参数检查
    assert src.dim() == 4, "src must be 4D [bsz, kv_heads, seq_len, hdim]"
    assert indices.dim(
    ) == 4, "indices must be 4D [bsz, kv_heads, num_groups, k]"
    assert indices_mask.shape == indices.shape, "mask shape must match indices"
    assert indices_mask.dtype == torch.bool, "mask must be bool tensor"
    assert src.device == indices.device == indices_mask.device, "All tensors must be on the same device"

    # 获取所有有效位置的坐标 (i,j,g,l)
    coords = indices_mask.nonzero(as_tuple=False)  # [num_valid, 4]

    # 提取对应的索引值
    idx_val = indices[indices_mask]  # [num_valid]

    # 从 src 中取出对应位置的向量
    i = coords[:, 0]
    j = coords[:, 1]
    dst = src[i, j, idx_val, :]  # [num_valid, hdim]

    return dst


def _v2_take_along_dim_with_mask_python(
    src: torch.Tensor,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    assert src.dim() == 4, "src must be 4D [bsz, kv_heads, seq_len, hdim]"
    assert indices.dim(
    ) == 4, "indices must be 4D [bsz, kv_heads, num_groups, k]"
    assert indices_mask.shape == indices.shape, "mask shape must match indices"
    assert indices_mask.dtype == torch.bool, "mask must be bool tensor"

    coords = indices_mask.nonzero(as_tuple=False)  # [num_valid,4] on GPU
    i = coords[:, 0]
    j = coords[:, 1]
    idx_val = indices[indices_mask]

    kv_heads, seq_len = src.shape[1], src.shape[2]
    linear_indices = (i * kv_heads * seq_len) + \
        (j * seq_len) + idx_val  # [num_valid]

    linear_indices_cpu = linear_indices.cpu()

    flat_src = src.view(-1, src.size(3))  # [bsz*kv_heads*seq_len, hdim]
    # dst_cpu = flat_src[linear_indices_cpu, :]  # [num_valid, hdim]

    dst_cpu = torch.ops.aten.index_select(flat_src, 0, linear_indices_cpu)

    return dst_cpu.to(indices.device)



def take_along_dim_with_mask_python(
    src: torch.Tensor,
    indices: torch.Tensor,
    indices_mask: torch.Tensor
) -> torch.Tensor:
    # 前置校验（生产环境可注释）
    assert src.is_contiguous(), "src must be contiguous memory"

    # 使用掩码直接计算一维偏移量
    kv_heads, seq_len = src.shape[1], src.shape[2]
    strides = torch.tensor([kv_heads*seq_len, seq_len, 1], dtype=torch.int64)

    # 三维坐标生成（保持四维掩码结构）
    bsz, kv_heads, num_groups, k = indices_mask.shape

    # 生成四维完整坐标（仅GPU显存允许时）
    coords_4d = indices_mask.nonzero(as_tuple=False)  # [num_valid,4]
    i = coords_4d[:, 0]  # bsz
    j = coords_4d[:, 1]  # kv_heads
    g = coords_4d[:, 2]  # group
    k_idx = coords_4d[:, 3]  # k

    # 直接索引四维张量
    idx_val = indices[i, j, g, k_idx]  # [num_valid]

    # 计算线性索引
    strides = torch.tensor([kv_heads*num_groups*k, num_groups*k, k, 1], dtype=torch.int64)
    linear_indices = (i * strides[0] +
                        j * strides[1] +
                        g * strides[2] +
                        k_idx * strides[3])

    # 第二层优化：零拷贝数据传输 --------------------------------------------
    # 使用DMA直接访问（需要CPU固定内存）
    pinned_src = src.pin_memory()
    linear_indices_cpu = linear_indices.cpu()

    # 第三层优化：SIMD加速CPU索引 -----------------------------------------
    # 使用内存视图+OpenMP并行
    with torch.autocast(device_type='cpu'):
        flat_src = pinned_src.view(-1, src.size(3))
        dst_cpu = torch.ops.aten.index_select(flat_src, 0, linear_indices_cpu)

    dst_gpu = dst_cpu.to(indices.device)
    torch.cuda.empty_cache()
    return dst_gpu


class TestTakeAlongDimGrouped(unittest.TestCase):
    def test_take_along_dim_grouped(self):
        pass


def only_indices_in_cpu(now_indices: torch.Tensor, new_indices: torch.Tensor):
    """
    Updates indices by replacing elements in now_indices with those from new_indices that are not already present,
    and generates a mask indicating which elements were replaced.

    Args:
        now_indices (torch.Tensor): Current indices in GPU, shape (bsz, num_heads, l, 1).
        new_indices (torch.Tensor): New indices required, shape (bsz, num_heads, l, 1).

    Returns:
        updated_indices (torch.Tensor): Updated indices with elements from new_indices where needed.
        dst_mask (torch.Tensor): Mask indicating which elements were replaced (1 if replaced, 0 otherwise).
    """
    bsz, num_heads, l, _ = now_indices.shape

    now_expanded = now_indices.view(bsz, num_heads, l, 1)
    new_expanded = new_indices.view(bsz, num_heads, 1, l)
    match_indices = (now_expanded == new_expanded)  # (N, M)

    now_in_new = ~match_indices.any(dim=3)  # (N, M)
    new_in_now = ~match_indices.any(dim=2)  # (N, M)

    updated_indices = now_indices.clone()
    updated_indices[now_in_new] = new_indices[new_in_now]

    return updated_indices, now_in_new

if __name__ == "__main__":
    bsz, kvhead, groups, seqlen, hdim = 2, 8, 4, 32, 64
    topk = 8
    now_indices = torch.topk(torch.randn(bsz, kvhead * groups, seqlen, 1), k=topk, dim=2).indices
    new_indices = torch.topk(torch.randn(bsz, kvhead * groups, seqlen, 1), k=topk, dim=2).indices

    updated_indices, dst_mask = only_indices_in_cpu(now_indices, new_indices)
    with np.printoptions(formatter={'int': '{:2d}'.format}):
        print("now", now_indices[0,0,:,0].cpu().numpy())
        print("new", new_indices[0,0,:,0].cpu().numpy())
        print("now", (dst_mask[0,0].int()).cpu().numpy())
        # print("new", new_in_now[0,0].int())
        print("   ", updated_indices[0,0,:,0].cpu().numpy())

    src = torch.randn(bsz, kvhead, seqlen, hdim, dtype=torch.float16)

    updated_indices = updated_indices.view(bsz, kvhead, groups, topk)
    dst_mask = dst_mask.view(bsz, kvhead, groups, topk)

    # help(take_along_dim_grouped.take_along_dim_with_mask)

    # (sum(dst_mask), hdim)
    # out = take_along_dim_grouped.take_along_dim_with_mask(
    #     src,
    #     updated_indices,
    #     dst_mask,
    # )
    out = _v2_take_along_dim_with_mask_python(
        src,
        updated_indices,
        dst_mask,
    )

    out_full = torch.zeros(bsz, kvhead, groups, topk, hdim, dtype=torch.float16)
    out_full[dst_mask] = out

    # check
    for i, j, k, m in itertools.product(
        range(bsz), range(kvhead), range(groups), range(topk)):

        if not dst_mask[i,j,k,m]:
            # assert is zero
            assert torch.norm(out_full[i,j,k,m]) == 0
            continue
        assert torch.allclose(out_full[i,j,k,m], src[i,j,updated_indices[i,j,k,m],:])

    print("pass")

#%%