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
        backend: mine::Backend::Cpu { workers: 2 },
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
        backend: mine::Backend::Cpu { workers: 2 },
        out: out.clone(),
    })
    .unwrap();

    let file_lines_after = std::fs::read_to_string(&out).unwrap().lines().count();
    assert_eq!(file_lines_after, hits.len() + hits2.len());
}

#[test]
fn unopenable_output_path_fails_before_mining() {
    // A directory that does not exist must surface an error immediately rather
    // than spawning workers that mine (potentially unbounded) against a writer
    // that already died opening the file.
    let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
    let err = mine::run(mine::MineOpts {
        deployer,
        words: words::parse_words("dead").unwrap(),
        positions: words::Positions::Ends,
        inner_min: 6,
        start: 0,
        count: None, // unbounded: only fails fast if the file opens up front
        backend: mine::Backend::Cpu { workers: 2 },
        out: std::path::PathBuf::from("/no/such/directory/hits.jsonl"),
    })
    .unwrap_err();
    assert!(
        err.to_string().contains("cannot open output file"),
        "unexpected error: {err}"
    );
}

#[test]
fn gpu_backend_requires_feature() {
    let dir = tempfile::tempdir().unwrap();
    let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
    let err = mine::run(mine::MineOpts {
        deployer,
        words: words::parse_words("dead").unwrap(),
        positions: words::Positions::Ends,
        inner_min: 6,
        start: 0,
        count: Some(1),
        backend: mine::Backend::Gpu(mine::GpuConfig::default()),
        out: dir.path().join("hits.jsonl"),
    })
    .unwrap_err();
    assert!(
        err.to_string().contains("--features gpu"),
        "unexpected error: {err}"
    );
}

fn hex32(salt: u128) -> String {
    let bytes = b20::salt_bytes(salt);
    let mut out = vec![0u8; 64];
    b20::hex_lower(&bytes, &mut out);
    String::from_utf8(out).unwrap()
}
