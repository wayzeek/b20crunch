# b20crunch

A vanity address miner for B20 tokens on Base, written in Rust: finds salts that
spell words in your token address.

<img width="1280" height="640" alt="social-card" src="https://github.com/user-attachments/assets/4e4d66b8-7610-4f89-b3e1-ce42edb6914e" />

Every B20 token lives at an address shaped like this:

```
0xB2 000000000000000000 <variant> <eighteen hex characters>
```

The first part is fixed by the standard. The last 18 characters are derived from
`keccak256(abi.encode(deployer, salt))`. You choose the salt. Which means
those 18 characters can spell something. Your ticker. Your product. `c0ffee`.
`deadbeef`. Whatever fits in hex.

```
deployer 0x1111111111111111111111111111111111111111, word c0ffee
salt:    2000763
address: 0xB200000000000000000000c0FFeeA7F58D19C4Ef
```

## How it works

`createB20` derives the token address from the caller and a salt, nothing else.
That has three pleasant consequences:

- **Salts are deployer-bound.** A salt mined for your account is worthless to
  anyone else. Share it, post it, tweet it: only your deployer can use it.
- **No front-running.** Copying your pending transaction from a different sender
  derives a completely different address.
- **One salt, everywhere.** The derivation has no chain id and covers both B20
  variants (ASSET and STABLECOIN), so one salt gives the same address on every
  network your deployer uses.

One caveat: the *deployer account itself* must call `createB20`. A multisig, a
proxy, or a deployer contract in between changes the sender and voids the salt.
Mine against whatever account will actually send the transaction.

The full derivation and the grinding math live in
[docs/how-it-works.md](docs/how-it-works.md).

## Install

```
git clone https://github.com/wayzeek/b20crunch
cd b20crunch && cargo build --release
```

or `cargo install --git https://github.com/wayzeek/b20crunch`. Source is the only
distribution; there is nothing to trust but the code in front of you.

## Mine

```
b20crunch mine --deployer 0xYourDeployer --words c0ffee,deadbeef
```

Words must be hex-expressible: `0-9 a-f`, with the time-honored substitutions
o=0, l/i=1, s=5, t=7, g=6, z=2. Placement defaults to either end of the window
(`--positions prefix|suffix|ends|any`). Hits stream to `hits.jsonl` and to your
terminal. The file is append only, so resuming a run with `--start` never
clobbers hits an earlier run already wrote. Run bounded with `--count`, resume
with `--start`, cross-check the results against the live factory with
`--verify`.

Expected time to a hit at 100 MH/s, both-ends placement. These are averages;
the search is memoryless, so your run may be lucky or unlucky:

| word length | expected salts | expected time |
|---|---|---|
| 6 | 8.4M | under a second |
| 7 | 134M | ~1 s |
| 8 | 2.1B | ~20 s |
| 9 | 34B | ~6 min |
| 10 | 550B | ~1.5 h |
| 11 | 8.8T | ~1 day |

Letter casing is not choosable: EIP-55 checksum casing falls out of the address
itself, so `deadbeef` may render as `dEAdbEef`. Look at the exact rendering
before you fall in love.

## GPU (OpenCL)

The default build is CPU-only. The GPU backend ships behind a Cargo feature:

```
cargo build --release --features gpu
./target/release/b20crunch mine --gpu --deployer 0xYourDeployer --words dead,beef
```

`--device N` picks a GPU when several are present (the error lists them), and
`--gpu-batch N` sets salts per dispatch for tuning. Everything else -- words,
positions, `--start` resume, JSONL output, `--verify` -- behaves exactly as on
the CPU, and a fixed salt range produces the identical hit set on either
backend. The kernel ships as OpenCL source inside the binary and is compiled
by your GPU driver at runtime; there is nothing precompiled to trust.

Requires a working OpenCL runtime: NVIDIA and AMD drivers include one, and
macOS has one built in. GPU rates will be added to the table above only as
measured on named hardware.

## Verify

```
b20crunch verify --deployer 0xYourDeployer --salt 123456 --expect 0xB20...
```

Read-only. Derives locally, cross-checks both variant addresses against the live
factory, and reports whether they are still unclaimed. Always verify before you
deploy; a (deployer, salt) pair can only be consumed once per network.

## Deploy

```
b20crunch deploy --deployer 0xYourDeployer --salt 123456 --expect 0xB20... \
                 --name "My Token" --symbol MTK
```

Dry-run by default: local derivation, factory derivation, availability, and a
gasless simulation, then it prints the exact transaction (decoded arguments and
raw calldata) and stops. Add `--send` to broadcast, with the deployer's key in
the `B20_DEPLOYER_KEY` environment variable. The key is read only for `--send`,
only from that variable, and the tool refuses to send if the key doesn't match
the deployer. Deploys the ASSET variant; nothing here touches STABLECOIN setup.

## A note on taste

Don't mine impersonation or trademark-lookalike addresses. Explorers flag them,
communities notice, and an address that spells someone else's name is worth less
than the electricity it took to find. Spell your own thing.
