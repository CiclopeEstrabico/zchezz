import struct

def check_l1_bias(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    
    # L1 bias starts at 28 + 799*256*4 = 818204
    l1b_offset = 818204
    # Read 256 floats
    biases = struct.unpack(f'<{256}f', data[l1b_offset:l1b_offset + 256*4])
    
    # Count how many are negative. Since ReLU is max(0, x), highly negative biases mean more sparsity.
    neg_count = sum(1 for b in biases if b < 0)
    avg_bias = sum(biases) / len(biases)
    
    print(f"File: {filename}")
    print(f"Negative biases: {neg_count} / 256 ({(neg_count/256)*100:.1f}%)")
    print(f"Average bias: {avg_bias:.4f}")
    print("-" * 30)

check_l1_bias(r'c:\nnue_checkpoints\engine\c\zchezz_v186E\nnue_weights.bin')
check_l1_bias(r'c:\nnue_checkpoints\engine\c\zchezz_v186F\nnue_weights.bin')
check_l1_bias(r'c:\nnue_checkpoints\engine\c\zchezz_v186H\nnue_weights.bin')
