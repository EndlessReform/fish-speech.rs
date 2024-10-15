use anyhow::Result;
use candle_core::cuda::cudarc;
use candle_core::{DType, Device, IndexOp, Module, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
// use fish_speech_core::models::text2semantic::DualARTransformer;
use fish_speech_core::models::text2semantic::{
    get_mask, precompute_freqs_cis, Attention, BaseModelArgs,
};

const USE_CUDA_GRAPH: bool = true;

fn cuda_graph(attn: &mut Attention, config: &BaseModelArgs, device: &Device) -> Result<()> {
    let cu_device = match &device {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let cu_stream = cu_device.cu_stream();
    const SEQLEN: usize = 1;
    let mask = get_mask(SEQLEN, device)?;
    let (cos_full, sin_full) = precompute_freqs_cis(config, device, DType::BF16)?;
    let freqs_cis = (&cos_full.i(0..SEQLEN)?, &sin_full.i(0..SEQLEN)?);

    let input_buffer = Var::from_tensor(&Tensor::zeros((1, SEQLEN, 1024), DType::BF16, &device)?)?;
    let output_buffer = Var::from_tensor(&Tensor::zeros((1, SEQLEN, 1024), DType::BF16, &device)?)?;

    println!("Got here");
    {
        // Warmup step: meaningless work to load kernels
        // load_ptx cannot be called while capturing the stream so we need this to happen
        // beforehand.
        let mask = get_mask(SEQLEN, device)?;
        let (cos_full, sin_full) = precompute_freqs_cis(config, device, DType::BF16)?;
        let freqs_cis = (&cos_full.i(0..SEQLEN)?, &sin_full.i(0..SEQLEN)?);

        let input = Tensor::zeros((1, SEQLEN, 1024), DType::BF16, &device)?;
        // let x = ffwd.forward(&input)?;
        let x = attn.forward(input_buffer.as_tensor(), &mask, freqs_cis)?;
        // output_buffer.set(&x)?;
        device.synchronize()?;
    }
    println!("Slow ffwd worked");
    let mask = get_mask(SEQLEN, device)?;
    let (cos_full, sin_full) = precompute_freqs_cis(config, device, DType::BF16)?;
    let canonical_good_result = output_buffer
        .as_tensor()
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
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
    println!("Beginning to capture stream");
    {
        // let x =
        //     Tensor::ones_like(input_buffer.as_tensor())?.broadcast_add(input_buffer.as_tensor())?;
        // output_buffer.set(&x)?;
        // let input = Tensor::zeros((1, SEQLEN, 1024), DType::BF16, &device)?;
        let freqs_cis = (&cos_full.i(0..SEQLEN)?, &sin_full.i(0..SEQLEN)?);

        // let mut x = input_buffer.as_tensor().clone();
        let x = Tensor::zeros((1, SEQLEN, 1024), DType::BF16, &device)?;
        let x = attn.forward(&x, &mask, freqs_cis)?;
        println!("Forward succeeded");
        output_buffer.set(&x)?;
    }
    println!("Recorded");
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
            //         attn.clear_cache();
            if let Err(err) = device.synchronize() {
                println!("err: {err:?}");
                Err(err)?;
            }
            println!("done syncing");
            // Update input for next run
            // input_buffer.set(&output_buffer.as_tensor())?;
            unsafe {
                cudarc::driver::sys::lib()
                    .cuGraphDestroy(cu_graph)
                    .result()?;
                cudarc::driver::sys::lib()
                    .cuGraphExecDestroy(cu_graph_e)
                    .result()?;
            }
        }
    } else {
        device.synchronize()?;
    }
    // // assert_eq!(
    //     output_buffer
    //         .as_tensor()
    //         .to_dtype(DType::F32)?
    //         .to_vec3::<f32>()?,
    //     canonical_good_result
    // );
    println!("Ended successfully!");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cfg = BaseModelArgs::fish_speech_1_2();

    // Create blank variables
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::BF16, &device);

    let mut ffwd = Attention::load(&vb, &cfg, false)?;

    cuda_graph(&mut ffwd, &cfg, &device)?;
    println!("Done");
    return Ok(());
}
