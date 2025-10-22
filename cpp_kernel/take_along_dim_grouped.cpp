
#include <ATen/Parallel.h>
#include <torch/extension.h>
// #include <xmmintrin.h>  // For _mm_prefetch

#include <cstring>
#include <vector>

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")

torch::Tensor take_along_dim_with_mask(const torch::Tensor& src,
                                       const torch::Tensor& indices,
                                       const torch::Tensor& indices_mask) {
  CHECK_CPU(src);
  CHECK_CONTIGUOUS(src);
  CHECK_CPU(indices);
  CHECK_CONTIGUOUS(indices);
  CHECK_CPU(indices_mask);
  CHECK_CONTIGUOUS(indices_mask);

  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(indices_mask.scalar_type() == torch::kBool, "mask must be bool");
  TORCH_CHECK(src.dim() == 4, "src [bsz, kv_heads, seq_len, hdim]");
  TORCH_CHECK(indices.dim() == 4,
              "indices [bsz, kv_heads, num_groups, k]");
  TORCH_CHECK(indices_mask.sizes().equals(indices.sizes()),
              "shape mask != indices");

  const int64_t bsz = src.size(0);
  const int64_t kv_heads = src.size(1);
  const int64_t seq_len = src.size(2);
  const int64_t hdim = src.size(3);
  const int64_t num_groups = indices.size(2);
  const int64_t K = indices.size(3);
  const int64_t H = kv_heads;
  const int64_t G = num_groups;

  std::vector<std::vector<int64_t>> counts(bsz,
                                           std::vector<int64_t>(kv_heads, 0));
  const bool* mask_ptr = indices_mask.data_ptr<bool>();

  at::parallel_for(0, bsz * H, 0, [&](int64_t start, int64_t end) {
    for (int64_t idx = start; idx < end; ++idx) {
      const int64_t i = idx / H;
      const int64_t j = idx % H;
      int64_t cnt = 0;
      const int64_t mask_base = i * H * G * K + j * G * K;
      for (int64_t g = 0; g < G; ++g) {
        const int64_t g_offset = mask_base + g * K;
        for (int64_t l = 0; l < K; ++l) {
          if (mask_ptr[g_offset + l]) cnt++;
        }
      }
      counts[i][j] = cnt;
    }
  });

  int64_t total = 0;
  for (const auto& row : counts) {
    for (int64_t val : row) {
      total += val;
    }
  }

  auto dst = torch::empty({total, hdim}, src.options());

  std::vector<std::vector<int64_t>> offsets(bsz,
                                            std::vector<int64_t>(kv_heads, 0));
  int64_t current_offset = 0;
  for (int64_t i = 0; i < bsz; ++i) {
    for (int64_t j = 0; j < kv_heads; ++j) {
      offsets[i][j] = current_offset;
      current_offset += counts[i][j];
    }
  }

  const int64_t* indices_ptr = indices.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    src.scalar_type(), "take_along_dim_with_mask", [&] {
        const scalar_t* src_ptr = src.data_ptr<scalar_t>();
        scalar_t* dst_ptr = dst.data_ptr<scalar_t>();
        const size_t chunk_bytes = hdim * sizeof(scalar_t);

        at::parallel_for(0, bsz * H, 0, [&](int64_t start, int64_t end) {
          for (int64_t idx = start; idx < end; ++idx) {
            const int64_t i = idx / H;
            const int64_t j = idx % H;
            const int64_t cnt = counts[i][j];
            if (cnt == 0) continue;

            const int64_t offset = offsets[i][j];
            const int64_t mask_base = i * H * G * K + j * G * K;
            const int64_t src_base = (i * H * seq_len + j * seq_len) * hdim;

            int64_t dst_pos = offset * hdim;
            int64_t copied = 0;

            for (int64_t g = 0; g < G; ++g) {
              const int64_t g_offset = mask_base + g * K;
              for (int64_t l = 0; l < K; ++l) {
                const int64_t mask_idx = g_offset + l;
                if (mask_ptr[mask_idx]) {
                  const int64_t idx_val = indices_ptr[mask_idx];
                  const int64_t src_pos = src_base + idx_val * hdim;
                  std::memcpy(
                    dst_ptr + dst_pos,
                    src_ptr + src_pos,
                    chunk_bytes
                  );
                  dst_pos += hdim;
                  copied++;
                }
              }
            }
          }
        });
      });

  return dst;
}

/**
  * indices select
  * Input
  * @dst (K, hdim): output tensor
  * @src (-1, hdim): source tensor
  * @indices (K): indices
  */
torch::Tensor d0_indices_select(
  torch::Tensor& dst,
  const torch::Tensor& src,
  const torch::Tensor& indices) {

  int64_t K = indices.size(0);
  int64_t hdim = src.size(1);

  // constexpr int64_t PREFETCH_DISTANCE = 2;

  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    src.scalar_type(), "d0_indices_select", [&] {
      const scalar_t* src_ptr = src.data_ptr<scalar_t>();
      scalar_t* dst_ptr = dst.data_ptr<scalar_t>();
      const int64_t* indices_ptr = indices.data_ptr<int64_t>();
      const int64_t block_size = hdim * sizeof(scalar_t);

      // for (int64_t i = 0; i < K; ++i) {
      //   const int64_t idx = indices_ptr[i];
      //   const int64_t src_pos = idx * hdim;
      //   std::memcpy(
      //     dst_ptr + i * hdim,
      //     src_ptr + src_pos,
      //     block_size
      //   );
      // }
      const int64_t chunk_size = std::max(
        (int64_t)4096, (K * 4) / (int64_t)at::get_num_threads()
      );

      at::parallel_for(0, K, chunk_size, [&](int64_t start, int64_t end) {
        scalar_t* _dst_ptr = dst_ptr + start * hdim;
        for (int64_t i = start; i < end; ++i) {
          const int64_t idx = indices_ptr[i];
          const int64_t src_pos = idx * hdim;
          std::memcpy(
            // dst_ptr + i * hdim,
            _dst_ptr,
            src_ptr + src_pos,
            block_size
          );
          _dst_ptr += hdim;
        }
      });
    }
  );

  return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "take_along_dim_with_mask", 
    &take_along_dim_with_mask,
      "Optimized masked index selection\n"
      "(CPU only)"
      "(CUDA not supported)"
      "Fetch elements from src according to indices and mask\n"
      "If mask is True, then the corresponding element is taken.\n"
      "If mask is False, then the corresponding element is ignored.\n"
      "Requires:\n"
      " - src: [bsz, kv_heads, seq_len, hdim] float type\n"
      " - indices: [bsz, kv_heads, num_groups, k] torch.Long\n"
      " - mask: [bsz, kv_heads, num_groups, k] bool\n"
      "Returns:\n"
      " - dst: [sum(mask), hdim]"
  );

  m.def(
    "d0_indices_select",
    &d0_indices_select,
      "Optimized index selection\n"
  );
}


