// b20crunch OpenCL kernel: one work-item = one salt of the 64-byte B20
// preimage (12 zero bytes | 20-byte deployer | 32-byte big-endian salt),
// one keccak-f1600 permutation, match the 18-nibble tail.
//
// The host passes the same padded state template the CPU kernel uses
// (lanes 0..8 hold the preimage with salt lanes 6,7 zero; lane 8 = 0x01
// keccak domain pad at byte 64; lane 16 = 0x80 final pad bit at byte 135).
// Keccak padding, NOT SHA3: the domain byte is 0x01, never 0x06.

typedef ulong u64;

__constant u64 RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
};

// rotation offsets and pi destinations by flat lane index (x + 5y),
// identical to the tables in src/kernel.rs
__constant uint RHO[25] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39,
    41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
};
__constant uint PI_DST[25] = {
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22,
    23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
};

static inline u64 bswap64(u64 v) {
    v = ((v & 0x00FF00FF00FF00FFUL) << 8) | ((v >> 8) & 0x00FF00FF00FF00FFUL);
    v = ((v & 0x0000FFFF0000FFFFUL) << 16) | ((v >> 16) & 0x0000FFFF0000FFFFUL);
    return (v << 32) | (v >> 32);
}

static void f1600(u64 a[25]) {
    for (int r = 0; r < 24; r++) {
        u64 c[5], d[5], b[25];
        for (int x = 0; x < 5; x++)
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++)
            d[x] = c[(x + 4) % 5] ^ rotate(c[(x + 1) % 5], (u64)1);
        for (int i = 0; i < 25; i++)
            a[i] ^= d[i % 5];
        for (int i = 0; i < 25; i++)
            b[PI_DST[i]] = rotate(a[i], (u64)RHO[i]);
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                a[x + 5 * y] = b[x + 5 * y]
                    ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
        a[0] ^= RC[r];
    }
}

// salt = (start_hi:start_lo) + off as a 128-bit add with carry; the host
// guarantees no salt in the batch wraps past u128::MAX and that
// batch_base + batch_len never wraps u64
static void window_of(
    __global const u64 *tmpl, u64 start_hi, u64 start_lo, u64 off,
    u64 *win_hi, u64 *win_lo)
{
    u64 st[25];
    for (int i = 0; i < 25; i++)
        st[i] = tmpl[i];
    u64 lo = start_lo + off;
    u64 hi = start_hi + (u64)(lo < start_lo); // carry into the high half
    // the 32-byte big-endian salt occupies preimage bytes 48..64: lanes 6
    // and 7 in little-endian lane order, hence the byte swaps
    st[6] = bswap64(hi);
    st[7] = bswap64(lo);
    f1600(st);
    // big-endian u128 view of the first 16 digest bytes; the top 72 bits
    // are the grindable tail
    *win_hi = bswap64(st[0]);
    *win_lo = bswap64(st[1]);
}

// test-only entry point: dump every window so the host can diff the ported
// permutation against the tiny-keccak reference
__kernel void window_dump(
    __global const u64 *tmpl,
    u64 start_hi, u64 start_lo, u64 batch_base, u64 batch_len,
    __global u64 *out)
{
    u64 gid = get_global_id(0);
    if (gid >= batch_len)
        return;
    u64 hi, lo;
    window_of(tmpl, start_hi, start_lo, batch_base + gid, &hi, &lo);
    out[2 * gid] = hi;
    out[2 * gid + 1] = lo;
}

// entries: 5 u64s per row (mask_hi, mask_lo, value_hi, value_lo,
// word_index << 32 | pos), position-major (prefix, suffix, inner) and
// longest-word-first within each class, so first match = the CPU winner.
// hits: 3 uints per record (salt offset in batch, word index, pos).
__kernel void mine(
    __global const u64 *tmpl,
    u64 start_hi, u64 start_lo, u64 batch_base, u64 batch_len,
    __global const u64 *entries, uint n_entries,
    __global uint *hits, volatile __global uint *counter, uint capacity)
{
    u64 gid = get_global_id(0);
    if (gid >= batch_len)
        return;
    u64 win_hi, win_lo;
    window_of(tmpl, start_hi, start_lo, batch_base + gid, &win_hi, &win_lo);
    for (uint e = 0; e < n_entries; e++) {
        __global const u64 *en = entries + 5 * (u64)e;
        if ((win_hi & en[0]) == en[2] && (win_lo & en[1]) == en[3]) {
            uint idx = atomic_inc(counter);
            // the counter may run past capacity; never write past the buffer
            if (idx < capacity) {
                hits[3 * idx] = (uint)gid;
                hits[3 * idx + 1] = (uint)(en[4] >> 32);
                hits[3 * idx + 2] = (uint)en[4];
            }
            return;
        }
    }
}
