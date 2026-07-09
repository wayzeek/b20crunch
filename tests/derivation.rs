use b20crunch::b20;

const DOC_DEPLOYER: &str = "0x1111111111111111111111111111111111111111";

// Vectors generated from the proven Python implementation and confirmed against
// the live factory on Base mainnet (getB20Address) on 2026-07-09:
//   cast call 0xB20f000000000000000000000000000000000000 \
//     "getB20Address(uint8,address,bytes32)(address)" <variant> <deployer> <salt32> \
//     --rpc-url https://mainnet.base.org
const VECTORS: &[(u128, &str, &str, &str)] = &[
    (
        0,
        "f043c50fe795c69f30",
        "0xb200000000000000000000f043c50fe795c69f30",
        "0xb200000000000000000001F043C50fe795C69f30",
    ),
    (
        1,
        "8eec1c9afb183a84aa",
        "0xb2000000000000000000008eec1c9AFB183a84aa",
        "0xB2000000000000000000018eec1C9afb183a84aa",
    ),
    (
        42,
        "c775cb6f4825b8f8a4",
        "0xB200000000000000000000C775cB6F4825b8f8A4",
        "0xb200000000000000000001C775cb6F4825b8f8a4",
    ),
];

#[test]
fn derivation_matches_factory_confirmed_vectors() {
    let deployer = b20::parse_address(DOC_DEPLOYER).unwrap();
    for (salt, tail_hex, asset, stable) in VECTORS {
        let tail = b20::tail(&deployer, *salt);
        let mut hex = [0u8; 18];
        b20::hex_lower(&tail, &mut hex);
        assert_eq!(std::str::from_utf8(&hex).unwrap(), *tail_hex);
        assert_eq!(b20::eip55(&b20::b20_address(&tail, 0)), *asset);
        assert_eq!(b20::eip55(&b20::b20_address(&tail, 1)), *stable);
    }
}

#[test]
fn address_shape() {
    let tail = [0xFFu8; 9];
    let a = b20::b20_address(&tail, 1);
    assert_eq!(a[0], 0xB2);
    assert_eq!(&a[1..10], &[0u8; 9]);
    assert_eq!(a[10], 1);
    assert_eq!(&a[11..], &tail);
}

#[test]
fn eip55_reference_vectors() {
    // From EIP-55 itself
    for s in [
        "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
        "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
        "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
        "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
    ] {
        let a = b20::parse_address(s).unwrap();
        assert_eq!(b20::eip55(&a), s);
    }
}

#[test]
fn salt_bytes_is_big_endian_32() {
    let s = b20::salt_bytes(0x0102);
    assert_eq!(&s[..30], &[0u8; 30]);
    assert_eq!(&s[30..], &[0x01, 0x02]);
}

#[test]
fn parse_address_rejects_garbage() {
    assert!(b20::parse_address("0x123").is_err());
    assert!(b20::parse_address("not an address").is_err());
    assert!(b20::parse_address("0xZZ11111111111111111111111111111111111111").is_err());
}
