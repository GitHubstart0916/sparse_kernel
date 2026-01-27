import torch
import sparse_kernel_extension

# decode
kHeadGroup = 2
kSparseTopK = 96
kSparseBlockSize = 64
# token_num = 1
batch_size = 4
seqlen_q_max = 8192 
topk_max_val = seqlen_q_max // kSparseBlockSize
device = torch.device('cuda:0')

def gen_topk(batch_size, topk_max_val):
    rand_scores = torch.rand((kHeadGroup, batch_size, topk_max_val))
    
    indices = rand_scores.topk(kSparseTopK, dim=-1, largest=False).indices
    
    ret = indices.sort(dim=-1).values.to(dtype=torch.int32).to(device=device)

    return ret

# topk_idx: [head_group, token_num, kSparseTopK]
topk_idx = gen_topk(batch_size, topk_max_val)
# block_table: [batch_size, seqlen_q_max]
block_table = torch.tensor([_ for _ in range(1, seqlen_q_max * batch_size + 1)], dtype=torch.int32, device=device).reshape(batch_size, seqlen_q_max)
# token_to_bs: [token_num]
token_to_bs = torch.arange(0, batch_size, dtype=torch.int32, device=device) 
# seqlen_q: [batch_size]
seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device) 


# warm up

for _ in range(10):
    sparse_kernel_extension.get_block_table(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )
    sparse_kernel_extension.get_block_table_v2(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )
    sparse_kernel_extension.get_block_table_v3(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )


try:
    out_block_table_v1 = sparse_kernel_extension.get_block_table(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )

    print("Kernel executed successfully.")
    print(f"Output shape: {out_block_table_v1.shape}") 

except Exception as e:
    print(f"Error during kernel execution: {e}")
    

try:  
    out_block_table_v2 = sparse_kernel_extension.get_block_table_v2(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )

    print("Kernel executed successfully.")
    print(f"Output shape: {out_block_table_v2.shape}") 

except Exception as e:
    print(f"Error during kernel execution: {e}")

try:
    out_block_table_v3 = sparse_kernel_extension.get_block_table_v3(
        topk_idx,
        block_table,
        token_to_bs,
        seqlen_q,
        seqlen_q
    )
    print("Kernel executed successfully.")
    print(f"Output shape: {out_block_table_v3.shape}") 

except Exception as e:
    print(f"Error during kernel execution: {e}")

if torch.allclose(out_block_table_v1, out_block_table_v2):
    print("Outputs from v2 are identical.")
else:
    print("Outputs from v2 differ.")
    
if torch.allclose(out_block_table_v1, out_block_table_v3):
    print("Outputs from v3 are identical.")
else:
    print("Outputs from v3 differ.")
    
    