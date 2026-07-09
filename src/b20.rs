use tiny_keccak::{Hasher, Keccak};

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut k = Keccak::v256();
    k.update(data);
    k.finalize(&mut out);
    out
}

pub fn parse_address(s: &str) -> Result<[u8; 20], String> {
    let h = s.strip_prefix("0x").unwrap_or(s);
    if h.len() != 40 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{s} is not a 20-byte hex address"));
    }
    let mut out = [0u8; 20];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
    }
    Ok(out)
}

/// 32-byte big-endian salt from a u128 (high 16 bytes zero).
pub fn salt_bytes(salt: u128) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[16..].copy_from_slice(&salt.to_be_bytes());
    s
}

/// First 9 bytes of keccak256(abi.encode(deployer, salt)).
pub fn tail(deployer: &[u8; 20], salt: u128) -> [u8; 9] {
    let mut pre = [0u8; 64];
    pre[12..32].copy_from_slice(deployer);
    pre[32..].copy_from_slice(&salt_bytes(salt));
    let h = keccak256(&pre);
    let mut t = [0u8; 9];
    t.copy_from_slice(&h[..9]);
    t
}

/// 0xB2 | 9 zero bytes | variant | 9-byte tail.
pub fn b20_address(tail: &[u8; 9], variant: u8) -> [u8; 20] {
    let mut a = [0u8; 20];
    a[0] = 0xB2;
    a[10] = variant;
    a[11..].copy_from_slice(tail);
    a
}

/// Lowercase hex, 2 chars per byte, into a caller-provided buffer.
pub fn hex_lower(bytes: &[u8], out: &mut [u8]) {
    for (i, b) in bytes.iter().enumerate() {
        out[2 * i] = HEX_CHARS[(b >> 4) as usize];
        out[2 * i + 1] = HEX_CHARS[(b & 0xF) as usize];
    }
}

/// EIP-55 checksummed address string.
pub fn eip55(addr: &[u8; 20]) -> String {
    let mut hex = [0u8; 40];
    hex_lower(addr, &mut hex);
    let h = keccak256(&hex);
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, &c) in hex.iter().enumerate() {
        let nibble = (h[i / 2] >> (if i % 2 == 0 { 4 } else { 0 })) & 0xF;
        out.push(if c.is_ascii_alphabetic() && nibble >= 8 {
            c.to_ascii_uppercase() as char
        } else {
            c as char
        });
    }
    out
}
