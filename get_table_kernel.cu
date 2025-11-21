#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 64;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

constexpr int kHeadGroup = 2;
constexpr int kSparseBlockSize = 64;
constexpr int kSparseTopK = 96;

/// @param topk_idx: [head_group, token_num, kSparseTopK] topk sparse block index
/// @param block_table: [batch_size, seqlen_q_max / page_size] page table mapping token idx to page idx
/// @param token_to_bs: [token_num] mapping from token idx to batch idx
/// @param token_pos_in_bs: [token_num] mapping from token idx to position in batch
/// @param seqlen_q: [batch_size] sequence length for each batch
/// @param out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize]
/// @param token_num
/// @param seqlen_q_max
/// @param page_size
__global__ void get_block_table_cuda(const int *topk_idx, const int *block_table, const int *token_to_bs,
                                     const int *token_pos_in_bs, const int *seqlen_q, int *out_block_table,
                                     const int seqlen_q_max, const int token_num, const int page_size = 1) {

  auto tidx = threadIdx.x;
  auto widx = tidx / WARP_SIZE;
  auto lane_id = tidx % WARP_SIZE;

  auto token_idx = blockIdx.x;
  auto bs = token_to_bs[token_idx];
  auto pos_in_bs = token_pos_in_bs[token_idx];
  auto seq_len = seqlen_q[bs];
  auto num_pages = kSparseBlockSize / page_size;

  for (int h = 0; h < kHeadGroup; h++) {
    for (int k = widx; k < kSparseTopK; k += WARPS_PER_BLOCK) {
      int sp_bidx = topk_idx[h * token_num * kSparseTopK + token_idx * kSparseTopK + k];
      if (sp_bidx < 0)
        continue;
      for (int i = lane_id; i < num_pages; i += WARP_SIZE) {
        int token_idx_in_batch = sp_bidx * kSparseBlockSize + i * page_size;
        auto token_valid = (token_idx_in_batch < seq_len) && (token_idx_in_batch < pos_in_bs);
        auto out_block_idx =
            token_idx * kHeadGroup * kSparseTopK * num_pages + h * kSparseTopK * num_pages + k * num_pages + i;
        out_block_table[out_block_idx] =
            token_valid ? kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch / page_size] + h : 0;
      }
    }
  }
}

torch::Tensor get_block_table_wrapper(const torch::Tensor &topk_idx,        // [head_group, token_num, kSparseTopK]
                                      const torch::Tensor &block_table,     // [batch_size, seqlen_q_max]
                                      const torch::Tensor &token_to_bs,     // [token_num]
                                      const torch::Tensor &token_pos_in_bs, // [token_num]
                                      const torch::Tensor &seqlen_q         // [batch_size]
) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = kSparseTopK * kSparseBlockSize;

  // 2. 验证输入张量形状
  TORCH_CHECK(topk_idx.sizes() == torch::IntArrayRef({kHeadGroup, token_num, kSparseTopK}), "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() == torch::IntArrayRef({batch_size, seqlen_q_max}), "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table = torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                                               topk_idx.options() // 继承 dtype 和 device
                                               )
                                      .contiguous();

  get_block_table_cuda<<<token_num, THREADS_PER_BLOCK>>>(topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
                                                         token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
                                                         seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(),
                                                         seqlen_q_max, token_num);

  return out_block_table;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_block_table", &get_block_table_wrapper, "Sparse Attention Block Table Getter (CUDA)");
}