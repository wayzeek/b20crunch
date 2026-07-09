use crate::{b20, mine};
use alloy::primitives::{address, Address, B256, U256};
use alloy::providers::ProviderBuilder;
use alloy::sol;
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
}

fn salt32(salt: u128) -> B256 {
    B256::from(U256::from(salt))
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
