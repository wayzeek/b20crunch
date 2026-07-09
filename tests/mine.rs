use b20crunch::{b20, mine, words};

#[test]
fn mined_hits_rederive_and_match_placement() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("hits.jsonl");
    let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
    let hits = mine::run(mine::MineOpts {
        deployer,
        words: words::parse_words("dead").unwrap(),
        positions: words::Positions::Ends,
        inner_min: 6,
        start: 0,
        count: Some(2_000_000),
        workers: 2,
        out: out.clone(),
    })
    .unwrap();

    // p = ~2/16^4 per salt -> ~61 expected hits in 2M salts
    assert!(hits.len() > 20, "suspiciously few hits: {}", hits.len());
    for h in &hits {
        let salt: u128 = h.salt.parse().unwrap();
        let tail = b20::tail(&deployer, salt);
        assert_eq!(h.asset_address, b20::eip55(&b20::b20_address(&tail, 0)));
        assert_eq!(
            h.stablecoin_address,
            b20::eip55(&b20::b20_address(&tail, 1))
        );
        match h.position.as_str() {
            "prefix" => assert!(h.tail.starts_with(&h.word)),
            "suffix" => assert!(h.tail.ends_with(&h.word)),
            other => panic!("unexpected position {other}"),
        }
        assert_eq!(h.salt_bytes32, format!("0x{}", hex32(salt)));
    }
    let file_lines = std::fs::read_to_string(&out).unwrap().lines().count();
    assert_eq!(file_lines, hits.len());

    // resuming with the same --out must append, not truncate: a second run
    // picking up where the first left off should leave both runs' hits in
    // the file, proving --start-based resume never clobbers earlier hits.
    let hits2 = mine::run(mine::MineOpts {
        deployer,
        words: words::parse_words("dead").unwrap(),
        positions: words::Positions::Ends,
        inner_min: 6,
        start: 2_000_000,
        count: Some(2_000_000),
        workers: 2,
        out: out.clone(),
    })
    .unwrap();

    let file_lines_after = std::fs::read_to_string(&out).unwrap().lines().count();
    assert_eq!(file_lines_after, hits.len() + hits2.len());
}

fn hex32(salt: u128) -> String {
    let bytes = b20::salt_bytes(salt);
    let mut out = vec![0u8; 64];
    b20::hex_lower(&bytes, &mut out);
    String::from_utf8(out).unwrap()
}
