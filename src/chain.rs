use crate::{b20, mine};
use alloy::network::EthereumWallet;
use alloy::primitives::{address, Address, B256, U256};
use alloy::providers::ProviderBuilder;
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use alloy::sol_types::SolValue;
use anyhow::Context;

pub const FACTORY: Address = address!("B20f000000000000000000000000000000000000");
pub const DEFAULT_RPC: &str = "https://mainnet.base.org";

sol! {
    #[sol(rpc)]
    interface IB20Factory {
        function getB20Address(uint8 variant, address deployer, bytes32 salt) external view returns (address);
        function isB20Initialized(address token) external view returns (bool);
        function createB20(uint8 variant, bytes32 salt, bytes params, bytes[] hooks) external returns (address token);
    }

    #[sol(rpc)]
    interface IB20Token {
        function name() external view returns (string);
        function symbol() external view returns (string);
        function decimals() external view returns (uint8);
    }

    /// The ASSET params tuple accepted by `createB20`. Declared via `sol!` (rather
    /// than a bare Rust tuple containing `u8`) because the resolved alloy_sol_types
    /// version does not implement `SolValue` for a bare `u8`/`U8` (it is reserved
    /// for the `bytes`/`bytesN` specialization); a `sol!`-generated struct encodes
    /// its `uint8` fields directly at the `SolType` level instead.
    struct AssetParams {
        uint8 variant;
        string name;
        string symbol;
        address deployer;
        uint8 decimals;
    }
}

fn salt32(salt: u128) -> B256 {
    B256::from(U256::from(salt))
}

/// abi.encode of the ASSET params tuple as a single dynamic value
/// (offset-prefixed), byte-identical to the proven production encoding.
pub fn encode_params(name: &str, symbol: &str, deployer: Address, decimals: u8) -> Vec<u8> {
    let params = AssetParams {
        variant: 1,
        name: name.to_string(),
        symbol: symbol.to_string(),
        deployer,
        decimals,
    };
    (params,).abi_encode_params()
}

fn provider(rpc: &str) -> anyhow::Result<impl alloy::providers::Provider + Clone> {
    // If this builder method differs in the resolved alloy version, check the
    // context7 alloy docs; on_http/connect_http are historical names for the same thing.
    // Likewise for every sol! call below: depending on the alloy version,
    // `.call().await?` decodes single-return functions either to the bare value
    // or to a one-field struct — if the compiler complains, extract `._0`
    // consistently everywhere rather than mixing styles.
    let url = rpc.parse().with_context(|| format!("bad RPC url: {rpc}"))?;
    Ok(ProviderBuilder::new().connect_http(url))
}

/// Derive locally, cross-check against the live factory, report availability.
/// `expect` compares against the ASSET address; returns false on mismatch.
pub async fn verify(
    rpc: &str,
    deployer: [u8; 20],
    salt: u128,
    expect: Option<[u8; 20]>,
) -> anyhow::Result<bool> {
    let p = provider(rpc)?;
    let f = IB20Factory::new(FACTORY, &p);
    let d = Address::from(deployer);
    let s = salt32(salt);

    let tail = b20::tail(&deployer, salt);
    let local_asset = Address::from(b20::b20_address(&tail, 0));
    let local_stable = Address::from(b20::b20_address(&tail, 1));

    let asset = f
        .getB20Address(0, d, s)
        .call()
        .await
        .with_context(|| format!("getB20Address(ASSET) via {rpc}"))?;
    let stable = f
        .getB20Address(1, d, s)
        .call()
        .await
        .with_context(|| format!("getB20Address(STABLECOIN) via {rpc}"))?;
    let asset_taken = f
        .isB20Initialized(asset)
        .call()
        .await
        .with_context(|| format!("isB20Initialized(ASSET) via {rpc}"))?;
    let stable_taken = f
        .isB20Initialized(stable)
        .call()
        .await
        .with_context(|| format!("isB20Initialized(STABLECOIN) via {rpc}"))?;

    println!("deployer:   {}", b20::eip55(&deployer));
    println!("salt:       {} (0x{})", salt, {
        let b = b20::salt_bytes(salt);
        let mut h = vec![0u8; 64];
        b20::hex_lower(&b, &mut h);
        String::from_utf8(h).unwrap()
    });
    println!(
        "asset:      {asset} (initialized: {asset_taken}, local derivation {})",
        if asset == local_asset {
            "matches"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "stablecoin: {stable} (initialized: {stable_taken}, local derivation {})",
        if stable == local_stable {
            "matches"
        } else {
            "MISMATCH"
        }
    );
    anyhow::ensure!(
        asset == local_asset && stable == local_stable,
        "local derivation disagrees with the factory; file a bug"
    );

    if let Some(e) = expect {
        let e = Address::from(e);
        if e == asset {
            println!("RESULT: MATCH");
        } else {
            eprintln!("RESULT: MISMATCH (expected {e})");
            return Ok(false);
        }
    }
    Ok(true)
}

/// Cross-check every mined hit against the factory (used by `mine --verify`).
pub async fn verify_hits(
    rpc: &str,
    deployer: [u8; 20],
    hits: &[mine::HitRecord],
) -> anyhow::Result<()> {
    let p = provider(rpc)?;
    let f = IB20Factory::new(FACTORY, &p);
    let d = Address::from(deployer);
    let mut mismatches = 0usize;
    for h in hits {
        let salt: u128 = h.salt.parse()?;
        let s = salt32(salt);
        let asset = f
            .getB20Address(0, d, s)
            .call()
            .await
            .with_context(|| format!("getB20Address(ASSET) via {rpc}"))?;
        let stable = f
            .getB20Address(1, d, s)
            .call()
            .await
            .with_context(|| format!("getB20Address(STABLECOIN) via {rpc}"))?;
        let taken = f
            .isB20Initialized(asset)
            .call()
            .await
            .with_context(|| format!("isB20Initialized(ASSET) via {rpc}"))?;
        let ok = asset.to_string().eq_ignore_ascii_case(&h.asset_address)
            && stable
                .to_string()
                .eq_ignore_ascii_case(&h.stablecoin_address);
        if !ok {
            mismatches += 1;
        }
        println!(
            "{} salt={} factory={} {}",
            h.word,
            h.salt,
            asset,
            match (ok, taken) {
                (false, _) => "DERIVATION MISMATCH",
                (true, true) => "already initialized!",
                (true, false) => "ok, available",
            }
        );
    }
    anyhow::ensure!(
        mismatches == 0,
        "{mismatches} hit(s) disagree with the factory derivation; file a bug"
    );
    Ok(())
}

pub struct DeployOpts {
    pub rpc: String,
    pub deployer: [u8; 20],
    pub salt: u128,
    pub expect: [u8; 20],
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub send: bool,
}

/// Pre-flight (always) and, with `send`, broadcast a B20 ASSET deployment.
pub async fn deploy(o: DeployOpts) -> anyhow::Result<()> {
    let d = Address::from(o.deployer);
    let expect = Address::from(o.expect);
    let s = salt32(o.salt);
    println!(
        "== pre-flight: {} ({}, {} decimals) at {} with salt {}",
        o.name, o.symbol, o.decimals, expect, o.salt
    );

    // 1. local derivation
    let tail = b20::tail(&o.deployer, o.salt);
    let local = Address::from(b20::b20_address(&tail, 0));
    anyhow::ensure!(
        local == expect,
        "local derivation {local} != expected {expect}"
    );
    println!("   local derivation ok");

    // 2-4. on-chain derivation, availability, simulation
    let p = provider(&o.rpc)?;
    let f = IB20Factory::new(FACTORY, &p);
    let onchain = f
        .getB20Address(0, d, s)
        .call()
        .await
        .with_context(|| format!("getB20Address(ASSET) via {}", o.rpc))?;
    anyhow::ensure!(
        onchain == expect,
        "factory derivation {onchain} != expected {expect}"
    );
    println!("   factory derivation ok");
    let taken = f
        .isB20Initialized(expect)
        .call()
        .await
        .with_context(|| format!("isB20Initialized(ASSET) via {}", o.rpc))?;
    anyhow::ensure!(!taken, "address already initialized");
    println!("   address free");

    let params = encode_params(&o.name, &o.symbol, d, o.decimals);
    let call = f.createB20(0, s, params.clone().into(), vec![]).from(d);
    let simulated = call
        .call()
        .await
        .with_context(|| format!("createB20 simulation reverted via {}", o.rpc))?;
    anyhow::ensure!(simulated == expect, "simulation returned {simulated}");
    println!("   dry-run ok (factory accepted the calldata)");

    // the exact transaction, shown before anything signs
    let calldata = call.calldata().clone();
    println!("   tx: to={FACTORY} value=0");
    println!(
        "   createB20(variant=0, salt={s}, params=({}, {:?}, {:?}, {d}, {}), hooks=[])",
        1, o.name, o.symbol, o.decimals
    );
    println!("   calldata: {calldata}");

    if !o.send {
        println!("== dry-run only. Re-run with --send to broadcast.");
        return Ok(());
    }

    let key = std::env::var("B20_DEPLOYER_KEY")
        .context("set B20_DEPLOYER_KEY to the deployer's private key to use --send")?;
    let signer: PrivateKeySigner = key
        .trim()
        .parse()
        .context("B20_DEPLOYER_KEY is not a valid key")?;
    anyhow::ensure!(
        signer.address() == d,
        "key belongs to {}, not the deployer {d}; a different sender derives a different address",
        signer.address()
    );
    let wallet = EthereumWallet::from(signer);
    let url = o
        .rpc
        .parse()
        .with_context(|| format!("bad RPC url: {}", o.rpc))?;
    let wp = ProviderBuilder::new().wallet(wallet).connect_http(url);
    let wf = IB20Factory::new(FACTORY, &wp);

    println!("== broadcasting");
    let receipt = wf
        .createB20(0, s, params.into(), vec![])
        .send()
        .await
        .with_context(|| format!("createB20 broadcast via {}", o.rpc))?
        .get_receipt()
        .await
        .with_context(|| format!("waiting for receipt via {}", o.rpc))?;
    println!(
        "   tx: {} (status: {})",
        receipt.transaction_hash,
        receipt.status()
    );
    anyhow::ensure!(receipt.status(), "transaction reverted");

    let init = f
        .isB20Initialized(expect)
        .call()
        .await
        .with_context(|| format!("isB20Initialized(ASSET) via {}", o.rpc))?;
    let t = IB20Token::new(expect, &p);
    let name = t
        .name()
        .call()
        .await
        .with_context(|| format!("name() via {}", o.rpc))?;
    let symbol = t
        .symbol()
        .call()
        .await
        .with_context(|| format!("symbol() via {}", o.rpc))?;
    let decimals = t
        .decimals()
        .call()
        .await
        .with_context(|| format!("decimals() via {}", o.rpc))?;
    println!("== deployed: initialized={init} name={name} symbol={symbol} decimals={decimals}");
    anyhow::ensure!(init, "token reports uninitialized after inclusion");
    anyhow::ensure!(
        name == o.name && symbol == o.symbol && decimals == o.decimals,
        "on-chain metadata disagrees with the request: name={name} symbol={symbol} decimals={decimals}"
    );
    let chain_id = alloy::providers::Provider::get_chain_id(&p)
        .await
        .with_context(|| format!("get_chain_id via {}", o.rpc))?;
    match chain_id {
        8453 => println!("   https://basescan.org/token/{expect}"),
        84532 => println!("   https://sepolia.basescan.org/token/{expect}"),
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_constant() {
        assert_eq!(
            FACTORY.to_string(),
            "0xB20f000000000000000000000000000000000000"
        );
    }

    #[test]
    fn params_encoding_matches_proven_golden() {
        // Golden generated 2026-07-09 with:
        //   cast abi-encode "f((uint8,string,string,address,uint8))" \
        //     '(1,"Test Token","TEST",0x1111111111111111111111111111111111111111,18)'
        // byte-identical to the encoding used by every production deploy.
        let golden = "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000e000000000000000000000000011111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000012000000000000000000000000000000000000000000000000000000000000000a5465737420546f6b656e0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000045445535400000000000000000000000000000000000000000000000000000000";
        let golden_bytes = {
            let h = golden.trim_start_matches("0x").replace('\\', "");
            (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect::<Vec<u8>>()
        };
        let enc = encode_params(
            "Test Token",
            "TEST",
            address!("1111111111111111111111111111111111111111"),
            18,
        );
        assert_eq!(enc, golden_bytes);
    }

    // Network test: run manually with `cargo test -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn live_factory_agrees_with_local_derivation() {
        let deployer =
            crate::b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let ok = verify(DEFAULT_RPC, deployer, 0, None).await.unwrap();
        assert!(ok);
    }
}
