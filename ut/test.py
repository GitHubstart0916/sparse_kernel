import torch
import sparse_kernel_extension


kHeadGroup = 2
kSparseTopK = 96
kSparseBlockSize = 64
BLOCK_SIZE = kSparseTopK * kSparseBlockSize
token_num = 8192
batch_size = 1
seqlen_q_max = 8192 

device = torch.device('cuda:0')

# topk_idx: [head_group, token_num, kSparseTopK]
topk_idx = torch.ones((kHeadGroup, token_num, kSparseTopK), dtype=torch.int32, device=device) * -1

# set valide blocks
topk_idx[0, 32, 0:2] = torch.tensor([ 0,  1 ], device=device, dtype=torch.int32) 
topk_idx[1, 32, 0:2] = torch.tensor([ 0,  1 ], device=device, dtype=torch.int32) 
topk_idx[1, 64, 0:2] = torch.tensor([ 0,  1 ], device=device, dtype=torch.int32) 
topk_idx[0, 1000, 0:10] = torch.tensor([ 0,  1,  5, 11, 14, 16, 17, 25, 26, 27], device=device, dtype=torch.int32)


# block_table: [batch_size, seqlen_q_max]
block_table = torch.tensor([_ for _ in range(1, seqlen_q_max * batch_size + 1)], dtype=torch.int32, device=device).reshape(batch_size, seqlen_q_max)

# token_to_bs: [token_num]
token_to_bs = torch.zeros((token_num,), dtype=torch.int32, device=device) 

# token_pos_in_bs: [token_num]
token_pos_in_bs = torch.tensor([_ for _ in range(1, token_num + 1)], dtype=torch.int32, device=device) 

# seqlen_q: [batch_size]
seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device) 


try:
    out_block_table = sparse_kernel_extension.get_block_table(
        topk_idx,
        block_table,
        token_to_bs,
        token_pos_in_bs,
        seqlen_q
    )

    print("Kernel executed successfully.")
    print(f"Output shape: {out_block_table.shape}") 

except Exception as e:
    print(f"Error during kernel execution: {e}")
    
    
    
#   check 32
assert (out_block_table[32, 0] != 0).sum().item() == 33
assert (out_block_table[32, 1] != 0).sum().item() == 33
assert torch.equal(out_block_table[32, 0, 0:33], block_table[0][:33] * 2)
assert torch.equal(out_block_table[32, 1, 0:33], block_table[0][:33] * 2 + 1)

# check 64
assert (out_block_table[64, 1] != 0).sum().item() == 65
assert torch.equal(out_block_table[64, 1, 0:65], block_table[0][:65] * 2 + 1)

#   check 1000
topk_blocks = [ 0,  1,  5, 11, 14, 16, 17, 25, 26, 27]
tokens = []
for b in topk_blocks:
    tokens.extend( [ _ for _ in range(b * kSparseBlockSize, (b + 1) * kSparseBlockSize)] )

tokens = [ t for t in tokens if t < token_num and t < 1001 ]

assert (out_block_table[1000, 0] != 0).sum().item() == len(tokens)
assert torch.equal(out_block_table[1000, 0, :len(tokens)], block_table[0][tokens] * 2)

print("All checks passed.")