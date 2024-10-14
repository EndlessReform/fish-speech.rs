use anyhow::Result;
use candle_core::cuda::cudarc;
use candle_core::{DType, Device, Tensor, Var};

const USE_CUDA_GRAPH: bool = true;

fn cuda_graph() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cu_device = match &device {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let cu_stream = cu_device.cu_stream();

    let input_buffer = Var::from_tensor(&Tensor::zeros(1024, DType::BF16, &device)?)?;
    let output_buffer = Var::from_tensor(&Tensor::zeros(1024, DType::BF16, &device)?)?;

    {
        // Warmup step: meaningless work to load kernels
        // load_ptx cannot be called while capturing the stream so we need this to happen
        // beforehand.
        let x =
            Tensor::ones_like(input_buffer.as_tensor())?.broadcast_add(input_buffer.as_tensor())?;
        let x = (x - 1f64)?;
        output_buffer.set(&x)?;
        device.synchronize()?;
    }
    if USE_CUDA_GRAPH {
        unsafe {
            cudarc::driver::sys::lib()
            .cuStreamBeginCapture_v2(
                *cu_stream,
                cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?
        }
    }
    {
        let x =
            Tensor::ones_like(input_buffer.as_tensor())?.broadcast_add(input_buffer.as_tensor())?;
        output_buffer.set(&x)?;
    }
    if USE_CUDA_GRAPH {
        let cu_graph: cudarc::driver::sys::CUgraph = unsafe {
            let mut cu_graph = std::mem::MaybeUninit::uninit();
            cudarc::driver::sys::lib()
                .cuStreamEndCapture(*cu_stream, cu_graph.as_mut_ptr())
                .result()?;
            cu_graph.assume_init()
        };
        let cu_graph_e: cudarc::driver::sys::CUgraphExec = unsafe {
            let mut cu_graph_e = std::mem::MaybeUninit::uninit();
            cudarc::driver::sys::lib()
                .cuGraphInstantiateWithFlags(cu_graph_e.as_mut_ptr(), cu_graph, 0)
                .result()?;
            cu_graph_e.assume_init()
        };
        println!("graph captured!");
        for i in 1..100 {
            println!("graph exec {i}");
            // Set out to result
            unsafe {
                cudarc::driver::sys::lib()
                    .cuGraphLaunch(cu_graph_e, *cu_stream)
                    .result()?
            }
            println!("sync");
            if let Err(err) = device.synchronize() {
                println!("err: {err:?}")
            }
            println!("done syncing");
            // Update input for next run
            input_buffer.set(&output_buffer.as_tensor())?;
        }
        unsafe {
            cudarc::driver::sys::lib()
                .cuGraphDestroy(cu_graph)
                .result()?;
            cudarc::driver::sys::lib()
                .cuGraphExecDestroy(cu_graph_e)
                .result()?;
        }
    } else {
        device.synchronize()?;
    }
    assert_eq!(
        output_buffer
            .as_tensor()
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?,
        vec![99u32; 1024]
    );
    Ok(())
}

fn main() -> Result<()> {
    cuda_graph()?;
    return Ok(());
}
