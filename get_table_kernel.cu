// #include <bits/stdc++.h>
// #include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
constexpr int kTokenNum = 8192;
constexpr int kBs = 1;
constexpr int kSeqlenQMax = 8192;
constexpr int kHeadGroup = 2;
// constexpr int kseqlenQ_max
constexpr int kSparseBlockSize = 64;
constexpr int kSparseTopK = 96;
// topk_idx: [head_group, token_num, kSparseTopK]: int32 [2, 8192, 96]
// block_table: [batch_size, seqlen_q_max]: int32 [1, 8192]
// token_to_bs: [token_num]: int32  [8192]
// token_pos_in_bs: [token_num]: int32 [8192]
// seqlen_q: [batch_size]: int32    [1]
// out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize]:
// int32 [2, 8192, 96 * 64] seqlen_q_max: int
__global__ void get_block_table_cuda(const int *topk_idx, const int *block_table,
                                const int *token_to_bs,
                                const int *token_pos_in_bs, const int *seqlen_q,
                                int *out_block_table, const int seqlen_q_max,
                                const int token_num) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= token_num) return;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];

  for (int h = 0; h < kHeadGroup; h++) {
    for (int i = 0; i < kSparseTopK * kSparseBlockSize; i++) {
      int sparse_block_idx =
          topk_idx[h * token_num * kSparseTopK + token_idx * kSparseTopK +
                   i / kSparseBlockSize];
      if (sparse_block_idx < 0)
        continue;
      int token_idx_in_batch =
          sparse_block_idx * kSparseBlockSize + (i % kSparseBlockSize);

      if (token_idx_in_batch < seqlen_q[bs] && token_idx_in_batch < pos_in_bs) {
        out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + h * kSparseTopK * kSparseBlockSize + i] =
            kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] + h;
      } else {
        out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + h * kSparseTopK * kSparseBlockSize + i] = 0;
      }
    }
  }
}



torch::Tensor get_block_table_wrapper(
    const torch::Tensor& topk_idx,        // [head_group, token_num, kSparseTopK]
    const torch::Tensor& block_table,     // [batch_size, seqlen_q_max]
    const torch::Tensor& token_to_bs,     // [token_num]
    const torch::Tensor& token_pos_in_bs, // [token_num]
    const torch::Tensor& seqlen_q         // [batch_size]
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
    

    torch::Tensor out_block_table = torch::zeros(
        {token_num, kHeadGroup, BLOCK_SIZE}, 
        topk_idx.options() // 继承 dtype 和 device
    ).contiguous();

  
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (token_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // 5. 调用 CUDA kernel
    get_block_table_cuda<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        topk_idx.data_ptr<int>(),
        block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(),
        token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(),
        out_block_table.data_ptr<int>(),
        seqlen_q_max,
        token_num
    );

    cudaDeviceSynchronize(); // 确保 CUDA kernel 执行完成

    return out_block_table;
}

// --- 4. PyTorch 扩展模块注册 ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_block_table", &get_block_table_wrapper, "Sparse Attention Block Table Getter (CUDA)");
}