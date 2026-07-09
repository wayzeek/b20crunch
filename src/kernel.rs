//! Specialized keccak for the miner's hot loop.
//!
//! The mining preimage is a fixed 64-byte single block: 12 zero bytes, the
//! 20-byte deployer, then the 32-byte salt. That means one keccak-f1600
//! permutation per candidate with a state template that only changes in two
//! lanes (the low 16 salt bytes), so the generic absorb/pad/squeeze machinery
//! of a streaming hasher is pure overhead here. This module keeps a
//! precomputed state template per deployer and exposes the first 16 output
//! bytes as a big-endian u128 whose top 72 bits are the grindable tail.
//!
//! Two permutation backends:
//! - a portable scalar keccak-f1600 (any platform),
//! - a 2-way NEON implementation using the ARMv8.2 SHA3 instructions
//!   (EOR3/RAX1/XAR/BCAX), selected at runtime, which hashes two salts per
//!   permutation call on Apple Silicon and other SHA3-capable ARM cores.
//!
//! Correctness is pinned by tests against the tiny-keccak reference in
//! `b20::tail` over random deployers and salts, plus the factory-confirmed
//! vectors in tests/derivation.rs which exercise this kernel through `mine`.

const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets by flat lane index (x + 5y).
const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, //
    36, 44, 6, 55, 20, //
    3, 10, 43, 25, 39, //
    41, 45, 15, 21, 8, //
    18, 2, 61, 56, 14,
];

/// pi destination by flat source index: (x, y) -> (y, 2x + 3y).
const PI_DST: [usize; 25] = [
    0, 10, 20, 5, 15, //
    16, 1, 11, 21, 6, //
    7, 17, 2, 12, 22, //
    23, 8, 18, 3, 13, //
    14, 24, 9, 19, 4,
];

/// Portable keccak-f1600 permutation.
pub fn f1600(a: &mut [u64; 25]) {
    for &rc in &RC {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for (i, lane) in a.iter_mut().enumerate() {
            *lane ^= d[i % 5];
        }
        let mut b = [0u64; 25];
        for i in 0..25 {
            b[PI_DST[i]] = a[i].rotate_left(RHO[i]);
        }
        for y in 0..5 {
            for x in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

/// Per-deployer keccak state template plus backend selection.
pub struct TailKernel {
    /// Lanes 0..8 hold the padded preimage (salt lanes 6 and 7 left zero),
    /// lane 8 is the 0x01 domain pad, lane 16 carries the final 0x80.
    template: [u64; 25],
    #[cfg(target_arch = "aarch64")]
    use_sha3: bool,
}

impl TailKernel {
    pub fn new(deployer: &[u8; 20]) -> TailKernel {
        let mut pre = [0u8; 64];
        pre[12..32].copy_from_slice(deployer);
        let mut template = [0u64; 25];
        for (i, chunk) in pre.chunks_exact(8).enumerate() {
            template[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        template[8] = 0x01; // keccak pad: 0x01 at byte 64
        template[16] = 0x8000_0000_0000_0000; // final pad bit: 0x80 at byte 135
        TailKernel {
            template,
            #[cfg(target_arch = "aarch64")]
            use_sha3: std::arch::is_aarch64_feature_detected!("sha3"),
        }
    }

    /// Salt lanes for the state: the 32-byte salt is big-endian u128 in bytes
    /// 48..64 of the preimage, i.e. lanes 6 and 7 in little-endian lane order.
    #[inline(always)]
    fn salt_lanes(salt: u128) -> (u64, u64) {
        (
            ((salt >> 64) as u64).swap_bytes(),
            (salt as u64).swap_bytes(),
        )
    }

    /// First 16 digest bytes as a big-endian u128; bits 127..=56 are the
    /// 18-nibble grindable window.
    #[inline]
    pub fn window(&self, salt: u128) -> u128 {
        let mut st = self.template;
        (st[6], st[7]) = Self::salt_lanes(salt);
        f1600(&mut st);
        ((st[0].swap_bytes() as u128) << 64) | st[1].swap_bytes() as u128
    }

    /// Two windows per call; on SHA3-capable aarch64 both salts share one
    /// 2-way vector permutation.
    #[inline]
    pub fn window2(&self, salt_a: u128, salt_b: u128) -> (u128, u128) {
        #[cfg(target_arch = "aarch64")]
        if self.use_sha3 {
            // SAFETY: use_sha3 is only set when runtime detection confirms
            // the sha3 target feature.
            return unsafe { neon::window2(&self.template, salt_a, salt_b) };
        }
        (self.window(salt_a), self.window(salt_b))
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::RC;
    use core::arch::aarch64::*;

    /// ROL(a ^ d, R) via XAR, which computes ROR(a ^ b, IMM).
    macro_rules! xar {
        ($a:expr, $d:expr, $r:literal) => {
            vxarq_u64::<{ 64 - $r }>($a, $d)
        };
    }

    /// 2-way keccak-f1600: vector lane 0 carries hash A, lane 1 hash B.
    #[target_feature(enable = "sha3")]
    unsafe fn f1600x2(a: &mut [uint64x2_t; 25]) {
        for &rc in &RC {
            // theta: column parities and the rotated cross-column mix
            let c0 = veor3q_u64(a[0], a[5], veor3q_u64(a[10], a[15], a[20]));
            let c1 = veor3q_u64(a[1], a[6], veor3q_u64(a[11], a[16], a[21]));
            let c2 = veor3q_u64(a[2], a[7], veor3q_u64(a[12], a[17], a[22]));
            let c3 = veor3q_u64(a[3], a[8], veor3q_u64(a[13], a[18], a[23]));
            let c4 = veor3q_u64(a[4], a[9], veor3q_u64(a[14], a[19], a[24]));
            let d0 = vrax1q_u64(c4, c1);
            let d1 = vrax1q_u64(c0, c2);
            let d2 = vrax1q_u64(c1, c3);
            let d3 = vrax1q_u64(c2, c4);
            let d4 = vrax1q_u64(c3, c0);

            // theta-apply + rho + pi fused: b[pi(i)] = ROL(a[i] ^ d[i%5], rho[i])
            let b0 = veorq_u64(a[0], d0);
            let b10 = xar!(a[1], d1, 1);
            let b20 = xar!(a[2], d2, 62);
            let b5 = xar!(a[3], d3, 28);
            let b15 = xar!(a[4], d4, 27);
            let b16 = xar!(a[5], d0, 36);
            let b1 = xar!(a[6], d1, 44);
            let b11 = xar!(a[7], d2, 6);
            let b21 = xar!(a[8], d3, 55);
            let b6 = xar!(a[9], d4, 20);
            let b7 = xar!(a[10], d0, 3);
            let b17 = xar!(a[11], d1, 10);
            let b2 = xar!(a[12], d2, 43);
            let b12 = xar!(a[13], d3, 25);
            let b22 = xar!(a[14], d4, 39);
            let b23 = xar!(a[15], d0, 41);
            let b8 = xar!(a[16], d1, 45);
            let b18 = xar!(a[17], d2, 15);
            let b3 = xar!(a[18], d3, 21);
            let b13 = xar!(a[19], d4, 8);
            let b14 = xar!(a[20], d0, 18);
            let b24 = xar!(a[21], d1, 2);
            let b9 = xar!(a[22], d2, 61);
            let b19 = xar!(a[23], d3, 56);
            let b4 = xar!(a[24], d4, 14);

            // chi: a[x] = b[x] ^ (~b[x+1] & b[x+2]) row-wise, BCAX(a,b,c) = a ^ (b & ~c)
            a[0] = vbcaxq_u64(b0, b2, b1);
            a[1] = vbcaxq_u64(b1, b3, b2);
            a[2] = vbcaxq_u64(b2, b4, b3);
            a[3] = vbcaxq_u64(b3, b0, b4);
            a[4] = vbcaxq_u64(b4, b1, b0);
            a[5] = vbcaxq_u64(b5, b7, b6);
            a[6] = vbcaxq_u64(b6, b8, b7);
            a[7] = vbcaxq_u64(b7, b9, b8);
            a[8] = vbcaxq_u64(b8, b5, b9);
            a[9] = vbcaxq_u64(b9, b6, b5);
            a[10] = vbcaxq_u64(b10, b12, b11);
            a[11] = vbcaxq_u64(b11, b13, b12);
            a[12] = vbcaxq_u64(b12, b14, b13);
            a[13] = vbcaxq_u64(b13, b10, b14);
            a[14] = vbcaxq_u64(b14, b11, b10);
            a[15] = vbcaxq_u64(b15, b17, b16);
            a[16] = vbcaxq_u64(b16, b18, b17);
            a[17] = vbcaxq_u64(b17, b19, b18);
            a[18] = vbcaxq_u64(b18, b15, b19);
            a[19] = vbcaxq_u64(b19, b16, b15);
            a[20] = vbcaxq_u64(b20, b22, b21);
            a[21] = vbcaxq_u64(b21, b23, b22);
            a[22] = vbcaxq_u64(b22, b24, b23);
            a[23] = vbcaxq_u64(b23, b20, b24);
            a[24] = vbcaxq_u64(b24, b21, b20);

            // iota
            a[0] = veorq_u64(a[0], vdupq_n_u64(rc));
        }
    }

    #[inline(always)]
    unsafe fn load_state(template: &[u64; 25], salt_a: u128, salt_b: u128) -> [uint64x2_t; 25] {
        let pair = |a: u64, b: u64| vld1q_u64([a, b].as_ptr());
        let mut st = [vdupq_n_u64(0); 25];
        for (i, &lane) in template.iter().enumerate() {
            st[i] = vdupq_n_u64(lane);
        }
        let (a6, a7) = super::TailKernel::salt_lanes(salt_a);
        let (b6, b7) = super::TailKernel::salt_lanes(salt_b);
        st[6] = pair(a6, b6);
        st[7] = pair(a7, b7);
        st
    }

    #[inline(always)]
    unsafe fn windows_of(st: &[uint64x2_t; 25]) -> (u128, u128) {
        let win = |l0: u64, l1: u64| ((l0.swap_bytes() as u128) << 64) | l1.swap_bytes() as u128;
        (
            win(vgetq_lane_u64::<0>(st[0]), vgetq_lane_u64::<0>(st[1])),
            win(vgetq_lane_u64::<1>(st[0]), vgetq_lane_u64::<1>(st[1])),
        )
    }

    #[target_feature(enable = "sha3")]
    pub unsafe fn window2(template: &[u64; 25], salt_a: u128, salt_b: u128) -> (u128, u128) {
        let mut st = load_state(template, salt_a, salt_b);
        f1600x2(&mut st);
        windows_of(&st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b20;

    /// Deterministic 128-bit xorshift-style generator for test inputs.
    struct Rng(u128);
    impl Rng {
        fn next(&mut self) -> u128 {
            // xorshift128 on the raw state; quality is irrelevant, coverage is
            self.0 ^= self.0 << 15;
            self.0 ^= self.0 >> 43;
            self.0 ^= self.0 << 29;
            self.0
        }
    }

    fn window_reference(deployer: &[u8; 20], salt: u128) -> u128 {
        // b20::tail is tiny-keccak; extend to 16 bytes via the same digest
        let mut pre = [0u8; 64];
        pre[12..32].copy_from_slice(deployer);
        pre[32..].copy_from_slice(&b20::salt_bytes(salt));
        let h = b20::keccak256(&pre);
        u128::from_be_bytes(h[..16].try_into().unwrap())
    }

    #[test]
    fn kernel_matches_tiny_keccak_reference() {
        let mut rng = Rng(0x8f0c_35bd_9a17_e2d4_6b01_44c8_7d5a_130f);
        for case in 0..2000 {
            let mut deployer = [0u8; 20];
            for b in deployer.iter_mut() {
                *b = rng.next() as u8;
            }
            let salt = match case % 5 {
                0 => 0,
                1 => u64::MAX as u128,
                2 => u128::MAX,
                3 => rng.next() & 0xFFFF_FFFF, // small realistic salts
                _ => rng.next(),
            };
            let kernel = TailKernel::new(&deployer);
            let expect = window_reference(&deployer, salt);
            assert_eq!(kernel.window(salt), expect, "scalar, salt {salt}");
        }
    }

    #[test]
    fn window2_agrees_with_window() {
        // On SHA3-capable hardware this cross-checks the NEON path against
        // the scalar path; elsewhere it still covers the pairing plumbing.
        let mut rng = Rng(0x1d84_a2f9_00c3_57be_e6f2_8b1d_4a90_c375);
        for _ in 0..2000 {
            let mut deployer = [0u8; 20];
            for b in deployer.iter_mut() {
                *b = rng.next() as u8;
            }
            let kernel = TailKernel::new(&deployer);
            let (sa, sb) = (rng.next(), rng.next());
            let (wa, wb) = kernel.window2(sa, sb);
            assert_eq!(wa, kernel.window(sa));
            assert_eq!(wb, kernel.window(sb));
        }
    }

    #[test]
    fn window_top_bits_are_the_tail() {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let kernel = TailKernel::new(&deployer);
        for salt in [0u128, 1, 42, 1 << 100] {
            let tail = b20::tail(&deployer, salt);
            let window = kernel.window(salt);
            assert_eq!(&window.to_be_bytes()[..9], &tail);
        }
    }
}
