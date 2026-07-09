//! OpenCL mining backend: device selection, kernel build, buffer
//! management, and the batched launch loop. Feature-gated behind `gpu`;
//! the default build never touches this module.

use crate::mine::{GpuConfig, Matcher};
use anyhow::anyhow;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::program::Program;
use std::sync::Arc;

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

#[allow(dead_code)] // fields are consumed as the kernel and launch loop land
pub struct GpuMiner {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
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
        let max_wg = device
            .max_work_group_size()
            .map_err(|e| anyhow!("device query failed: {e}"))?;
        let max_alloc = device
            .max_mem_alloc_size()
            .map_err(|e| anyhow!("device query failed: {e}"))?;
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
        // size-within-device-limits: the hit buffer is the only allocation
        // that scales with config (batch is already capped at the u32
        // counter/offset width, which also bounds the global work size on
        // any 32-bit size_t device)
        let hit_buf_bytes = 3 * capacity as u64 * 4;
        anyhow::ensure!(
            hit_buf_bytes <= max_alloc,
            "hit capacity {capacity} needs a {hit_buf_bytes}-byte buffer, over the device's max allocation of {max_alloc} bytes"
        );
        eprintln!(
            "gpu: {name} (max work-group {max_wg}, max alloc {} MB)",
            max_alloc >> 20
        );
        Ok(GpuMiner {
            device,
            context,
            queue,
            program,
            max_wg,
            max_alloc,
            batch,
            capacity,
            deployer,
            matcher,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{b20, words};

    pub(super) fn test_miner(cfg: &GpuConfig) -> anyhow::Result<GpuMiner> {
        let deployer = b20::parse_address("0x1111111111111111111111111111111111111111").unwrap();
        let w = words::parse_words("dead").unwrap();
        let matcher = Arc::new(Matcher::new(&w, words::Positions::Ends, 6));
        GpuMiner::new(cfg, deployer, matcher)
    }

    #[test]
    fn builds_on_the_local_device() {
        let miner = test_miner(&GpuConfig::default()).unwrap();
        assert!(miner.max_wg >= 1);
        assert_eq!(miner.capacity, 1 << 16);
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
