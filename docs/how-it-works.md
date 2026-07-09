# How B20 addresses are derived, and how the grind works

The factory assigns every B20 token its address deterministically. Nothing about
the deployment transaction matters except two inputs: who calls `createB20`, and
the salt they pass.

## The derivation

A B20 address is 20 bytes assembled by rule, not by hashing init code. Byte 0 is
`0xB2`. Bytes 1 through 9 are zero. Byte 10 is the variant: `0x00` for ASSET,
`0x01` for STABLECOIN. Bytes 11 through 19 are the tail:

```
tail = keccak256(abi.encode(deployer, salt))[0..9]
```

`abi.encode(address, uint256)` produces a 64-byte preimage: 12 zero bytes, the
20-byte deployer, then the 32-byte big-endian salt. Hash it, keep the first 9
bytes, and the address is fully determined.

Since the first byte is `0xB2` and the next nine are zero, every B20 address
renders as `0xB2 000000000000000000 <variant>` followed by 18 hex characters of
tail. Those 18 characters are the only part you can influence, and you
influence them by trying salts.

Note what is absent from the preimage: no chain id, no init code hash, no `0xff`
prefix. This is not CREATE2, even though it rhymes with it. The factory computes
this address and deploys to it; the derivation is its own scheme.

## What follows from the preimage

Three properties fall straight out of those 64 bytes.

**Salts bind to one deployer.** The deployer is hashed into the tail, so a salt
mined for your account produces garbage for anyone else. You can post a salt in
public before using it and nobody can take the address from you. Front-running
is not mitigated, it is structurally absent: an attacker copying your pending
transaction is a different sender and lands at a different address.

**One salt works everywhere.** With no chain id in the preimage, the same
(deployer, salt) pair gives the same address on every network the factory
exists on. Mine once, deploy anywhere.

**Both variants come free.** The variant byte sits outside the hash, so one
tail serves both the ASSET and STABLECOIN addresses. Grinding for one is
grinding for both.

## The math of finding a word

Each salt is an independent draw of 18 uniform hex characters. A word of length
L matches a fixed position with probability 16^-L, so the expected number of
salts to a hit at a single end is 16^L. Allowing both ends roughly halves that
(the exact union term subtracts the both-ends overlap; it only matters for
words short enough to fit twice). Allowing the word anywhere in the window
gives 19 - L placements, which is why `--positions any` finds long words 3 to
5x faster than `--positions ends`. Very short words are the exception: below
the inner-placement minimum the miner keeps them at the ends, where they still
read as intentional instead of drowning mid-string.

Concretely: a 6-char word at either end costs about 8.4M salts on average, an
8-char word about 2.1B, a 10-char word about 550B. Each extra character
multiplies the work by 16. The full window, 18 characters, is one placement at
16^18 and out of CPU reach; the practical ceiling on ordinary hardware is
around 10 or 11.

The search is memoryless. Expected counts are averages over many runs, not a
progress bar: a lucky run hits a 9-char word in minutes and an unlucky one
takes triple the average. Past salts tried tell you nothing about how far away
the next hit is.

## Casing is not yours to choose

EIP-55 checksumming derives the upper or lower case of each letter from a hash
of the address itself. `deadbeef` may come out `dEAdbEef`. The miner reports
the exact checksummed rendering of every hit; read it before you commit to a
word whose look you care about.

## From hit to deployment

A hit is a claim about math, not about the chain, so verify it against the live
factory before relying on it: `b20crunch verify` recomputes the address
locally, asks the factory for its own answer, and checks the address is still
unclaimed. Each (deployer, salt) pair can be consumed once per network. Then
`b20crunch deploy` runs the same checks plus a gasless simulation and prints
the exact transaction it would send; nothing is broadcast until you add
`--send`.

The words themselves must survive the hex alphabet: `0-9 a-f`, stretched by the
usual substitutions o=0, l/i=1, s=5, t=7, g=6, z=2. `c0ffee`, `deadbeef`,
`0dd101` all qualify. If your word does not fit, a shorter fragment of it
usually does, and every character you drop makes the grind 16x cheaper.
