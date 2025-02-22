use candle_core::{bail, CustomOp1, Layout, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RepeatKV {
    n_rep: usize,
}

impl CustomOp1 for RepeatKV {
    fn name(&self) -> &'static str {
        "repeat_kv"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        bail!("Not implemented. Please just use repeat_kv directly");
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        _storage: &candle_core::MetalStorage,
        _layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        bail!("Not implemented");
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
        };
        use candle_core::cuda_backend::{kernel_name, Map1, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S {
            n_repeats: usize,
        }
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType + candle_core::cuda::cudarc::driver::ValidAsZeroBits>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => bail!("Input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let input_el = layout.shape().elem_count();
                let output_el = input_el * self.n_repeats;
                let (_bsz, n_local_heads, seqlen, head_dim) = layout.shape().dims4()?;

                let cfg = LaunchConfig {
                    grid_dim: (seqlen as u32, n_local_heads as u32, self.n_repeats as u32),
                    block_dim: (head_dim as u32, 1, 1),
                    shared_mem_bytes: 0,
                };

                let func = dev
                    .get_or_load_func(&kernel_name::<T>("repeat_kv"), candle_gqa_kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(output_el) }.w()?;
                let params = (&src, &dst, n_local_heads, self.n_repeats, seqlen, head_dim);

                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(dst)
            }
        }

        use candle_core::backend::BackendStorage;

        let (bsz, n_local_heads, seqlen, head_dim) = l1.shape().dims4()?;
        if bsz != 1 {
            bail!("Only implemented for single batch. Repeat_interleave will be almost as fast at higher bsz");
        }

        let dev = s1.device();
        let slice = S {
            n_repeats: self.n_rep,
        }
        .map(&s1.slice, dev, l1)?;
        let dst = candle_core::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };

        Ok((
            dst,
            (bsz, n_local_heads * self.n_rep, seqlen, head_dim).into(),
        ))
    }
}

pub fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if !xs.is_contiguous() {
        bail!("Input must be contiguous");
    }
    xs.apply_op1(RepeatKV { n_rep })
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    fn repeat_kv_slow(xs: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
            // Using cat is faster than a broadcast as it avoids going through a potentially
            // strided copy.
            // https://github.com/huggingface/candle/pull/2043
            Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
        }
    }

    #[test]
    fn test_accuracy_seq1() {
        let device = Device::new_cuda(0).unwrap();
        let key_states = Tensor::randn(0f32, 1f32, (1, 2, 1, 64), &device).unwrap();
        let n_rep = 8usize;

        let ref_tensor = repeat_kv_slow(key_states.clone(), n_rep)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let kernel_tensor = repeat_kv(&key_states, n_rep).unwrap().squeeze(0).unwrap();
        assert_eq!(ref_tensor.shape(), kernel_tensor.shape());
        assert_eq!(ref_tensor.layout(), kernel_tensor.layout());
        assert_eq!(
            ref_tensor.to_vec3::<f32>().unwrap(),
            kernel_tensor.to_vec3::<f32>().unwrap()
        )
    }

    #[test]
    fn test_accuracy_seqn() {
        let device = Device::new_cuda(0).unwrap();
        let key_states = Tensor::randn(0f32, 1f32, (1, 2, 166, 64), &device).unwrap();
        let n_rep = 8usize;

        let ref_tensor = repeat_kv_slow(key_states.clone(), n_rep)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let kernel_tensor = repeat_kv(&key_states, n_rep).unwrap().squeeze(0).unwrap();
        assert_eq!(ref_tensor.shape(), kernel_tensor.shape());
        assert_eq!(ref_tensor.layout(), kernel_tensor.layout());
        assert_eq!(
            ref_tensor.to_vec3::<f32>().unwrap(),
            kernel_tensor.to_vec3::<f32>().unwrap()
        )
    }
}
