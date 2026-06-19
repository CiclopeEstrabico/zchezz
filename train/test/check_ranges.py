import struct

def check_weights(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        
    off = 28 # magic + 5*4
    L1_SZ = 799 * 256
    
    l1w = struct.unpack(f'<{L1_SZ}f', data[off : off + L1_SZ*4])
    off += L1_SZ*4
    l1b = struct.unpack(f'<{256}f', data[off : off + 256*4])
    off += 256*4
    
    L2_SZ = 256 * 64
    l2w = struct.unpack(f'<{L2_SZ}f', data[off : off + L2_SZ*4])
    off += L2_SZ*4
    l2b = struct.unpack(f'<{64}f', data[off : off + 64*4])
    off += 64*4
    
    l3w = struct.unpack(f'<{64}f', data[off : off + 64*4])
    off += 64*4
    l3b = struct.unpack('<f', data[off : off + 4])
    
    print(f"L1W min: {min(l1w):.4f}, max: {max(l1w):.4f}")
    print(f"L1B min: {min(l1b):.4f}, max: {max(l1b):.4f}")
    print(f"L2W min: {min(l2w):.4f}, max: {max(l2w):.4f}")
    print(f"L2B min: {min(l2b):.4f}, max: {max(l2b):.4f}")
    print(f"L3W min: {min(l3w):.4f}, max: {max(l3w):.4f}")
    print(f"L3B: {l3b[0]:.4f}")

check_weights(r'c:\nnue_checkpoints\engine\c\zchezz_v188E\nnue_weights.bin')
