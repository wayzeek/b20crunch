//! OpenCL mining backend: device selection, kernel build, buffer
//! management, and the batched launch loop. Feature-gated behind `gpu`;
//! the default build never touches this module.

use crate::mine::{GpuConfig, Matcher, RawHit};
use anyhow::anyhow;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{
    Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY,
};
use opencl3::program::Program;
use opencl3::types::{cl_uint, cl_ulong, CL_BLOCKING};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

const KERNEL_SRC: &str = include_str!("gpu/kernel.cl");

/// Salts per dispatch when --gpu-batch is not given: large enough to keep a
/// discrete GPU busy, small enough that Ctrl-C lands within a batch or two.
const DEFAULT_BATCH: u64 = 1 << 24;

// GpuMiner crosses a thread boundary in the launch loop; fail the build here
// rather than in that later change if opencl3's types ever stop being Send.
const _: fn() = || {
    fn check<T: Send>() {}
    check::<GpuMiner>();
};

/// One batch's outcome: hit records `(offset_in_batch, word_index, pos_code)`
/// or an overflow that the caller must shrink and rerun.
pub enum BatchOutcome {
    Hits(Vec<(u32, u32, u32)>),
    Overflow,
}

// Not every field is read in the release mining path: max_wg/max_alloc are
// held for the deferred perf-tuning pass, dump_kernel is test plumbing, and
// device/program document what the queue and kernels were built from.
#[allow(dead_code)]
pub struct GpuMiner {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    dump_kernel: Kernel,
    mine_kernel: Kernel,
    tmpl_buf: Buffer<cl_ulong>,
    entries_buf: Buffer<cl_ulong>,
    hits_buf: Buffer<cl_uint>,
    counter_buf: Buffer<cl_uint>,
    n_entries: u32,
    max_wg: usize,
    max_alloc: u64,
    batch: u64,
    capacity: u32,
    deployer: [u8; 20],
    matcher: Arc<Matcher>,
}

impl GpuMiner {
    /// Selects a device, builds the kernel, and validates the config.
    /// Loud and early: a missing runtime, an ambiguous device choice, or an
    /// unbuildable kernel is an error here, before any mining state exists.
    pub fn new(
        cfg: &GpuConfig,
        deployer: [u8; 20],
        matcher: Arc<Matcher>,
    ) -> anyhow::Result<GpuMiner> {
        // device-independent config checks first: a bad --gpu-batch must
        // report as such even on a machine with no OpenCL runtime at all
        let batch = cfg.batch.unwrap_or(DEFAULT_BATCH);
        // the hit counter and the record offset field are u32: a batch must
        // never be able to wrap them
        anyhow::ensure!(
            (1..=u32::MAX as u64).contains(&batch),
            "--gpu-batch must be between 1 and {}",
            u32::MAX
        );
        let capacity =
            u32::try_from(cfg.capacity).map_err(|_| anyhow!("hit capacity must fit in u32"))?;
        // records are 3 uints wide and indexed as 3 * idx inside the kernel;
        // keep that multiply inside u32 range
        anyhow::ensure!(
            (1..=u32::MAX / 3).contains(&capacity),
            "hit capacity must be between 1 and {}",
            u32::MAX / 3
        );
        let ids = get_all_devices(CL_DEVICE_TYPE_GPU).map_err(|e| {
            anyhow!("OpenCL enumeration failed (is an OpenCL runtime/ICD installed?): {e}")
        })?;
        anyhow::ensure!(!ids.is_empty(), "no OpenCL GPU devices found");
        let listing = || -> String {
            ids.iter()
                .enumerate()
                .map(|(i, &id)| {
                    let name = Device::new(id).name().unwrap_or_else(|_| "?".into());
                    format!("  --device {i}: {name}")
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let idx = match cfg.device {
            Some(i) => {
                anyhow::ensure!(
                    i < ids.len(),
                    "--device {i} does not exist; available:\n{}",
                    listing()
                );
                i
            }
            None if ids.len() == 1 => 0,
            None => anyhow::bail!(
                "multiple GPUs found; pick one with --device:\n{}",
                listing()
            ),
        };
        let device = Device::new(ids[idx]);
        let name = device
            .name()
            .map_err(|e| anyhow!("device query failed: {e}"))?;
        let context =
            Context::from_device(&device).map_err(|e| anyhow!("context creation failed: {e}"))?;
        // create_default calls clCreateCommandQueue: correct for OpenCL 1.2
        // devices (Apple); the WithProperties variants are 2.0+ only.
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|e| anyhow!("queue creation failed: {e}"))?;
        let program = Program::create_and_build_from_source(&context, KERNEL_SRC, "")
            .map_err(|e| anyhow!("OpenCL kernel build failed on {name}; driver build log:\n{e}"))?;
        let dump_kernel = Kernel::create(&program, "window_dump")
            .map_err(|e| anyhow!("kernel handle failed: {e}"))?;
        let tmpl_host: Vec<cl_ulong> = crate::kernel::TailKernel::new(&deployer)
            .template()
            .to_vec();
        // SAFETY: COPY_HOST_PTR reads tmpl_host during creation only
        let tmpl_buf = unsafe {
            Buffer::<cl_ulong>::create(
                &context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                25,
                tmpl_host.as_ptr() as *mut c_void,
            )
        }
        .map_err(|e| anyhow!("template buffer failed: {e}"))?;
        let max_wg = device
            .max_work_group_size()
            .map_err(|e| anyhow!("device query failed: {e}"))?;
        let max_alloc = device
            .max_mem_alloc_size()
            .map_err(|e| anyhow!("device query failed: {e}"))?;
        // size-within-device-limits: the hit buffer is the only allocation
        // that scales with config (batch is already capped at the u32
        // counter/offset width, which also bounds the global work size on
        // any 32-bit size_t device)
        let hit_buf_bytes = 3 * capacity as u64 * 4;
        anyhow::ensure!(
            hit_buf_bytes <= max_alloc,
            "hit capacity {capacity} needs a {hit_buf_bytes}-byte buffer, over the device's max allocation of {max_alloc} bytes"
        );
        let mine_kernel =
            Kernel::create(&program, "mine").map_err(|e| anyhow!("kernel handle failed: {e}"))?;
        let entries = matcher.gpu_entries();
        let flat: Vec<cl_ulong> = entries
            .iter()
            .flat_map(|e| {
                [
                    e.mask_hi,
                    e.mask_lo,
                    e.value_hi,
                    e.value_lo,
                    ((e.word as u64) << 32) | e.pos as u64,
                ]
            })
            .collect();
        let n_entries = entries.len() as u32;
        // SAFETY: COPY_HOST_PTR reads flat during creation; the other
        // buffers are device-populated
        let (entries_buf, hits_buf, counter_buf) = unsafe {
            (
                Buffer::<cl_ulong>::create(
                    &context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    flat.len(),
                    flat.as_ptr() as *mut c_void,
                )
                .map_err(|e| anyhow!("entries buffer failed: {e}"))?,
                Buffer::<cl_uint>::create(
                    &context,
                    CL_MEM_WRITE_ONLY,
                    3 * capacity as usize,
                    std::ptr::null_mut(),
                )
                .map_err(|e| anyhow!("hits buffer failed: {e}"))?,
                Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, std::ptr::null_mut())
                    .map_err(|e| anyhow!("counter buffer failed: {e}"))?,
            )
        };
        eprintln!(
            "gpu: {name} (max work-group {max_wg}, max alloc {} MB)",
            max_alloc >> 20
        );
        Ok(GpuMiner {
            device,
            context,
            queue,
            program,
            dump_kernel,
            mine_kernel,
            tmpl_buf,
            entries_buf,
            hits_buf,
            counter_buf,
            n_entries,
            max_wg,
            max_alloc,
            batch,
            capacity,
            deployer,
            matcher,
        })
    }

    /// Hash and match `batch_len` salts from `start + batch_base`. Never
    /// emits a partial batch: capacity overruns discard the whole batch.
    /// Crate-private: callers must uphold the no-wrap range preconditions
    /// that mine_loop's clamp arithmetic guarantees.
    pub(crate) fn run_batch(
        &mut self,
        start: u128,
        batch_base: u64,
        batch_len: u64,
    ) -> anyhow::Result<BatchOutcome> {
        debug_assert!(batch_len >= 1 && batch_len <= u32::MAX as u64);
        // SAFETY: buffer sizes match the slice lengths; blocking reads fence
        // the in-order queue between launch and readback.
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.counter_buf, CL_BLOCKING, 0, &[0u32], &[])
                .map_err(|e| anyhow!("counter reset failed: {e}"))?;
            ExecuteKernel::new(&self.mine_kernel)
                .set_arg(&self.tmpl_buf)
                .set_arg(&((start >> 64) as cl_ulong))
                .set_arg(&(start as cl_ulong))
                .set_arg(&batch_base)
                .set_arg(&batch_len)
                .set_arg(&self.entries_buf)
                .set_arg(&self.n_entries)
                .set_arg(&self.hits_buf)
                .set_arg(&self.counter_buf)
                .set_arg(&self.capacity)
                .set_global_work_size(batch_len as usize)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| anyhow!("mine launch failed: {e}"))?;
            let mut count = [0 as cl_uint];
            self.queue
                .enqueue_read_buffer(&self.counter_buf, CL_BLOCKING, 0, &mut count, &[])
                .map_err(|e| anyhow!("counter readback failed: {e}"))?;
            if count[0] > self.capacity {
                return Ok(BatchOutcome::Overflow);
            }
            let n = count[0] as usize;
            let mut raw = vec![0 as cl_uint; 3 * n];
            if n > 0 {
                self.queue
                    .enqueue_read_buffer(&self.hits_buf, CL_BLOCKING, 0, &mut raw, &[])
                    .map_err(|e| anyhow!("hit readback failed: {e}"))?;
            }
            Ok(BatchOutcome::Hits(
                raw.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect(),
            ))
        }
    }

    /// Re-derive a GPU hit on the CPU and require agreement; the writer only
    /// ever sees CPU-derived values, so a kernel bug can produce a loud error
    /// but never a wrong JSONL line.
    fn check_hit(
        &self,
        salt: u128,
        window: u128,
        word_idx: u32,
        pos_code: u32,
    ) -> anyhow::Result<RawHit> {
        let Some((pos, word)) = self.matcher.find(window) else {
            anyhow::bail!("GPU hit at salt {salt} rejected by the CPU matcher: kernel bug");
        };
        let gpu_word = self.matcher.word(word_idx as usize).ok_or_else(|| {
            anyhow!("GPU hit at salt {salt} has out-of-range word index {word_idx}: kernel bug")
        })?;
        anyhow::ensure!(
            word == gpu_word && pos.code() == pos_code,
            "GPU hit disagrees with CPU at salt {salt}: gpu ({gpu_word}, {pos_code}) vs cpu ({word}, {})",
            pos.code(),
        );
        let mut tail = [0u8; 9];
        tail.copy_from_slice(&window.to_be_bytes()[..9]);
        Ok(RawHit {
            word: word.to_string(),
            pos,
            salt,
            tail,
        })
    }

    /// Batched launch loop: the GPU producer feeding the shared hit channel.
    /// Runs until the count, the u64 offset horizon (matching the CPU
    /// dispatch counter), the end of the u128 salt space, or Ctrl-C.
    /// Crate-private because it traffics in RawHit, which never leaves the
    /// crate (a `pub fn` here would trip the private_interfaces lint).
    pub(crate) fn mine_loop(
        mut self,
        start: u128,
        count: Option<u64>,
        tx: mpsc::Sender<RawHit>,
        scanned: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let cpu = crate::kernel::TailKernel::new(&self.deployer);
        // adaptive batch: halve on overflow, creep back up on clean batches,
        // so a hit-dense word list settles near its sustainable batch size
        // instead of re-overflowing every round
        let mut cur = self.batch;
        let mut base: u64 = 0;
        while !stop.load(Ordering::Relaxed) {
            let remaining = match count {
                Some(c) if base >= c => break,
                Some(c) => c - base,
                None if base == u64::MAX => break,
                None => u64::MAX - base,
            };
            let Some(first) = start.checked_add(base as u128) else {
                break;
            };
            let requested = cur.min(remaining);
            // clamp so no salt in the batch wraps past u128::MAX (the CPU
            // worker clamps identically)
            let mut len = ((requested as u128 - 1).min(u128::MAX - first) + 1) as u64;
            // shrink-and-rerun: an overflowed batch is discarded whole and
            // retried smaller; capacity >= 1 makes a size-1 batch always clean
            let accepted = loop {
                match self.run_batch(start, base, len)? {
                    BatchOutcome::Overflow => len = (len / 2).max(1),
                    BatchOutcome::Hits(hits) => {
                        for (off, word_idx, pos_code) in hits {
                            // the host validates everything the kernel
                            // reports; an offset outside the batch would
                            // otherwise re-derive some unrelated salt
                            anyhow::ensure!(
                                (off as u64) < len,
                                "GPU hit offset {off} outside the {len}-salt batch: kernel bug"
                            );
                            let salt = first + off as u128;
                            let window = cpu.window(salt);
                            let hit = self.check_hit(salt, window, word_idx, pos_code)?;
                            if tx.send(hit).is_err() {
                                return Ok(()); // writer gone: shutting down
                            }
                        }
                        break len;
                    }
                }
            };
            // only an accepted batch advances progress: an overflowed run
            // contributes nothing to the scanned count or resume arithmetic
            scanned.fetch_add(accepted, Ordering::Relaxed);
            cur = if accepted < requested {
                accepted
            } else {
                cur.saturating_mul(2).min(self.batch)
            };
            if first + (accepted - 1) as u128 == u128::MAX {
                break; // the u128 salt space itself is exhausted
            }
            base += accepted;
        }
        Ok(())
    }

    /// Hash `batch_len` salts from `start + batch_base` and return each
    /// window; test plumbing for diffing the ported permutation against
    /// the tiny-keccak reference (the window_dump kernel entry it drives
    /// ships unconditionally, so any build can be probed by a test build).
    #[cfg(test)]
    pub(crate) fn dump_windows(
        &mut self,
        start: u128,
        batch_base: u64,
        batch_len: u64,
    ) -> anyhow::Result<Vec<u128>> {
        let n = batch_len as usize;
        let mut out = vec![0 as cl_ulong; 2 * n];
        // SAFETY: buffer sizes match the slice lengths used below; the
        // blocking read fences the in-order queue.
        unsafe {
            let out_buf = Buffer::<cl_ulong>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                2 * n,
                std::ptr::null_mut(),
            )
            .map_err(|e| anyhow!("window buffer failed: {e}"))?;
            ExecuteKernel::new(&self.dump_kernel)
                .set_arg(&self.tmpl_buf)
                .set_arg(&((start >> 64) as cl_ulong))
                .set_arg(&(start as cl_ulong))
                .set_arg(&batch_base)
                .set_arg(&batch_len)
                .set_arg(&out_buf)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| anyhow!("window_dump launch failed: {e}"))?;
            self.queue
                .enqueue_read_buffer(&out_buf, CL_BLOCKING, 0, &mut out, &[])
                .map_err(|e| anyhow!("window readback failed: {e}"))?;
        }
        Ok(out
            .chunks_exact(2)
            .map(|c| ((c[0] as u128) << 64) | c[1] as u128)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{b20, words};

    /// Device index for test runs: hosts with several GPUs set
    /// B20CRUNCH_TEST_DEVICE; single-GPU hosts leave it unset and exercise
    /// the auto-selection path.
    fn test_device() -> Option<usize> {
        std::env::var("B20CRUNCH_TEST_DEVICE").ok().map(|v| {
            v.parse()
                .expect("B20CRUNCH_TEST_DEVICE must be a device index")
        })
    }

    pub(super) fn test_cfg() -> GpuConfig {
        GpuConfig {
            device: test_device(),
            ..Default::default()
        }
    }

    pub(super) fn test_miner(cfg: &GpuConfig) -> anyhow::Result<GpuMiner> {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let w = words::parse_words("dead").unwrap();
        let matcher = Arc::new(Matcher::new(&w, words::Positions::Ends, 6));
        GpuMiner::new(cfg, deployer, matcher)
    }

    #[test]
    fn builds_on_the_local_device() {
        let miner = test_miner(&test_cfg()).unwrap();
        assert!(miner.max_wg >= 1);
        assert_eq!(miner.capacity, 1 << 16);
    }

    #[test]
    fn gpu_windows_match_the_cpu_kernel() {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let cpu = crate::kernel::TailKernel::new(&deployer);
        let mut miner = test_miner(&test_cfg()).unwrap();
        // boundary salts from the spec: u64 carry, u64::MAX + 1, the end of
        // the u128 space, plus a mid-range value and a nonzero batch_base
        for &(start, base, len) in &[
            (0u128, 0u64, 256u64),
            ((u64::MAX as u128) - 128, 0, 256), // crosses the u64 carry
            ((u64::MAX as u128) + 1, 0, 64),
            (u128::MAX - 63, 0, 64), // last salt is exactly u128::MAX
            (0x1234_5678_9abc_def0_1122_3344_5566_7788u128, 0, 128),
            (42, 1 << 20, 128), // batch_base offsets add before the carry
        ] {
            let wins = miner.dump_windows(start, base, len).unwrap();
            assert_eq!(wins.len(), len as usize);
            for (i, w) in wins.iter().enumerate() {
                let salt = start + base as u128 + i as u128;
                assert_eq!(*w, cpu.window(salt), "salt {salt}");
            }
        }
    }

    #[test]
    fn rejects_a_nonexistent_device_index_with_a_listing() {
        let cfg = GpuConfig {
            device: Some(99),
            ..Default::default()
        };
        let err = test_miner(&cfg).map(|_| ()).unwrap_err().to_string();
        assert!(
            err.contains("--device 99 does not exist"),
            "unexpected: {err}"
        );
        assert!(err.contains("--device 0:"), "listing missing: {err}");
    }

    /// Overlap-heavy config: 2-nibble words in every position class.
    fn overlap_miner(capacity: usize) -> (GpuMiner, Arc<Matcher>) {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let w = words::parse_words("de,ad").unwrap();
        let matcher = Arc::new(Matcher::new(&w, words::Positions::Any, 2));
        let cfg = GpuConfig {
            capacity,
            ..test_cfg()
        };
        (
            GpuMiner::new(&cfg, deployer, matcher.clone()).unwrap(),
            matcher,
        )
    }

    #[test]
    fn batch_hits_equal_the_cpu_matcher() {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let (mut miner, matcher) = overlap_miner(1 << 17);
        let tk = crate::kernel::TailKernel::new(&deployer);
        let len = 200_000u64;
        let hits = match miner.run_batch(0, 0, len).unwrap() {
            BatchOutcome::Hits(h) => h,
            BatchOutcome::Overflow => panic!("unexpected overflow"),
        };
        let mut expect = std::collections::HashSet::new();
        for salt in 0..len as u128 {
            if let Some((pos, word)) = matcher.find(tk.window(salt)) {
                expect.insert((salt, word.to_string(), pos.code()));
            }
        }
        assert!(
            expect.len() > 10_000,
            "test range too sparse: {}",
            expect.len()
        );
        let got: std::collections::HashSet<_> = hits
            .iter()
            .map(|&(off, w, p)| {
                (
                    off as u128,
                    matcher.word(w as usize).unwrap().to_string(),
                    p,
                )
            })
            .collect();
        assert_eq!(got.len(), hits.len(), "duplicate GPU hit records");
        assert_eq!(expect, got);
    }

    #[test]
    fn overfull_batch_reports_overflow() {
        let (mut miner, _) = overlap_miner(4);
        match miner.run_batch(0, 0, 65_536).unwrap() {
            BatchOutcome::Overflow => {}
            BatchOutcome::Hits(h) => panic!("expected overflow, got {} hits", h.len()),
        }
    }

    #[test]
    fn tiny_batch_under_capacity_is_clean() {
        // capacity 1 can never overflow a single-salt batch: at most one hit
        let (mut miner, _) = overlap_miner(1);
        for salt in 0..64u64 {
            match miner.run_batch(salt as u128, 0, 1).unwrap() {
                BatchOutcome::Hits(h) => assert!(h.len() <= 1),
                BatchOutcome::Overflow => panic!("size-1 batch cannot overflow capacity 1"),
            }
        }
    }

    #[test]
    fn mine_loop_reports_hits_through_the_channel() {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let w = words::parse_words("dead").unwrap();
        let matcher = Arc::new(Matcher::new(&w, words::Positions::Ends, 6));
        let miner = GpuMiner::new(&test_cfg(), deployer, matcher).unwrap();
        let (tx, rx) = mpsc::channel();
        let scanned = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        miner
            .mine_loop(0, Some(2_000_000), tx, scanned.clone(), stop)
            .unwrap();
        let hits: Vec<RawHit> = rx.try_iter().collect();
        assert!(hits.len() > 20, "suspiciously few hits: {}", hits.len());
        assert_eq!(scanned.load(Ordering::Relaxed), 2_000_000);
        for h in &hits {
            let tail = b20::tail(&deployer, h.salt);
            assert_eq!(h.tail, tail, "salt {}", h.salt);
        }
    }

    #[test]
    fn rejects_zero_and_oversized_batch() {
        let err = test_miner(&GpuConfig {
            batch: Some(0),
            ..Default::default()
        })
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(err.contains("--gpu-batch"), "unexpected: {err}");
        let err = test_miner(&GpuConfig {
            batch: Some(1 << 33),
            ..Default::default()
        })
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(err.contains("--gpu-batch"), "unexpected: {err}");
    }
}
