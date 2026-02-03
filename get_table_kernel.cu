// #include <bits/stdc++.h>
// #include <cuda_runtime.h>
#include "static_switch.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
// constexpr int kTokenNum = 8192;
// constexpr int kBs = 1;
// constexpr int kSeqlenQMax = 8192;
constexpr int kHeadGroup = 2;
// // constexpr int kseqlenQ_max
constexpr int kSparseBlockSize = 64;
// constexpr int kSparseTopK = 96;
constexpr int kTopkPerBlock = 16;
// constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
// topk_idx: [head_group, token_num, kSparseTopK]: int32 [2, 8192, 96]
// block_table: [batch_size, seqlen_q_max]: int32 [1, 8192]
// token_to_bs: [token_num]: int32  [8192]
// token_pos_in_bs: [token_num]: int32 [8192]
// seqlen_q: [batch_size]: int32    [1]
// out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize]:
// int32 [2, 8192, 96 * 64] seqlen_q_max: int
template <int kSparseTopK>
__global__ void
get_block_table_cuda_v1(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= token_num)
    return;
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
        out_block_table[token_idx * kHeadGroup * kSparseTopK *
                            kSparseBlockSize +
                        h * kSparseTopK * kSparseBlockSize + i] =
            kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] +
            h;
      } else {
        out_block_table[token_idx * kHeadGroup * kSparseTopK *
                            kSparseBlockSize +
                        h * kSparseTopK * kSparseBlockSize + i] = 0;
      }
    }
  }
}

// 1 thread calc 64 element of out_block_table
// This allows topk_idx to be read once and all corresponding
// out_block_table elements calculated, reducing memory access
template <int kSparseTopK>
__global__ void
get_block_table_cuda_v2(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  int token_idx =
      (blockIdx.x * blockDim.x + threadIdx.x) / (kSparseTopK * kHeadGroup);
  if (token_idx >= token_num)
    return;
  int head_group_idx =
      ((blockIdx.x * blockDim.x + threadIdx.x) / kSparseTopK) % kHeadGroup;
  int topk_idx_in_head = (blockIdx.x * blockDim.x + threadIdx.x) % kSparseTopK;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];
  int seqlen_q_bs = seqlen_q[bs];
  int sparse_block_idx = topk_idx[head_group_idx * token_num * kSparseTopK +
                                  token_idx * kSparseTopK + topk_idx_in_head];

  if (sparse_block_idx < 0)
    return;
  for (int i = 0; i < kSparseBlockSize; i++) {

    int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + i;

    if (token_idx_in_batch < seqlen_q_bs && token_idx_in_batch < pos_in_bs) {
      out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
                      head_group_idx * kSparseTopK * kSparseBlockSize +
                      topk_idx_in_head * kSparseBlockSize + i] =
          kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] +
          head_group_idx;
    } else {
      out_block_table[token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
                      head_group_idx * kSparseTopK * kSparseBlockSize +
                      topk_idx_in_head * kSparseBlockSize + i] = 0;
    }
  }
}

// opt for decode
// 1 thread calc 1 element of out_block_table
// block size 1024
// smem 1024 / 64 = 16

template <int kSparseTopK>
__global__ void
get_block_table_cuda_v3(const int *topk_idx, const int *block_table,
                        const int *token_to_bs, const int *token_pos_in_bs,
                        const int *seqlen_q, int *out_block_table,
                        const int seqlen_q_max, const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  // calc 16 topk -> 1024 output
  __shared__ int topk_idx_share[kTopkPerBlock];
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  if (threadIdx.x < kTopkPerBlock) {
    topk_idx_share[tidx] = topk_idx[bidx * kTopkPerBlock + tidx];
  }

  __syncthreads();

  const int head_group_idx = (bidx / kBlockPerTokenHead) / token_num;
  const int token_idx = (bidx / kBlockPerTokenHead) % token_num;
  const int topk_idx_in_head =
      bidx % kBlockPerTokenHead * kTopkPerBlock + tidx / kSparseBlockSize;

  const int sparse_block_idx = topk_idx_share[tidx / kSparseBlockSize];

  const int token_idx_src =
      sparse_block_idx * kSparseBlockSize + tidx % kSparseBlockSize;
  const int token_idx_dst =
      token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
      head_group_idx * kSparseTopK * kSparseBlockSize +
      topk_idx_in_head * kSparseBlockSize + tidx % kSparseBlockSize;

  const int bs = token_to_bs[token_idx];
  const int pos_in_bs = token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];

  if (token_idx_src < seqlen_q_bs && token_idx_src < pos_in_bs) {
    out_block_table[token_idx_dst] =
        kHeadGroup * block_table[bs * seqlen_q_max + token_idx_src] +
        head_group_idx;
  } else {
    out_block_table[token_idx_dst] = 0;
  }
}

torch::Tensor get_block_table_v1_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 256;
  const int NUM_BLOCKS =
      (token_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v1<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

torch::Tensor get_block_table_v2_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 1024;
  const int NUM_BLOCKS =
      (token_num * kHeadGroup * topk + THREADS_PER_BLOCK - 1) /
      THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v2<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

torch::Tensor get_block_table_v3_wrapper(
    const torch::Tensor &topk_idx,    // [head_group, token_num, kSparseTopK]
    const torch::Tensor &block_table, // [batch_size, seqlen_q_max]
    const torch::Tensor &token_to_bs, // [token_num]
    const torch::Tensor &token_pos_in_bs, // [token_num]
    const torch::Tensor &seqlen_q,        // [batch_size]
    const int topk) {

  TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
  TORCH_CHECK(topk_idx.dtype() == torch::kInt, "All inputs must be int32");

  int token_num = topk_idx.size(1);
  int seqlen_q_max = block_table.size(1);
  const int batch_size = block_table.size(0);
  const int BLOCK_SIZE = topk * kSparseBlockSize;

  TORCH_CHECK(topk_idx.sizes() ==
                  torch::IntArrayRef({kHeadGroup, token_num, topk}),
              "topk_idx shape incorrect");
  TORCH_CHECK(block_table.sizes() ==
                  torch::IntArrayRef({batch_size, seqlen_q_max}),
              "block_table shape incorrect");
  TORCH_CHECK(token_to_bs.size(0) == token_num, "token_to_bs size incorrect");

  torch::Tensor out_block_table =
      torch::zeros({token_num, kHeadGroup, BLOCK_SIZE},
                   topk_idx.options() // 继承 dtype 和 device
                   )
          .contiguous();

  const int THREADS_PER_BLOCK = 1024;
  const int NUM_BLOCKS = (token_num * kHeadGroup * topk * kSparseBlockSize +
                          THREADS_PER_BLOCK - 1) /
                         THREADS_PER_BLOCK;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VALUE_SPLITS_SWITCH(topk, kSparseTopK, [&]() {
    auto kernel = get_block_table_cuda_v3<kSparseTopK>;
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        topk_idx.data_ptr<int>(), block_table.data_ptr<int>(),
        token_to_bs.data_ptr<int>(), token_pos_in_bs.data_ptr<int>(),
        seqlen_q.data_ptr<int>(), out_block_table.data_ptr<int>(), seqlen_q_max,
        token_num);
  });

  // cudaDeviceSynchronize();

  return out_block_table;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_block_table_v1", &get_block_table_v1_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
  m.def("get_block_table_v2", &get_block_table_v2_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
  m.def("get_block_table_v3", &get_block_table_v3_wrapper,
        "Sparse Attention Block Table Getter (CUDA)");
}