use ec_gpu::GpuName;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use log::{debug, error};
use rust_gpu_tools::{program_closures, Device, Program};
use std::ops::AddAssign;
use std::sync::{Arc, RwLock};
use yastl::Scope;

use crate::{
    error::{EcError, EcResult},
    threadpool::Worker,
};

/// On the GPU, the exponents are split into windows, this is the maximum number of such windows.
const MAX_WINDOW_SIZE: usize = 10;
/// In CUDA this is the number of blocks per grid (grid size).
const LOCAL_WORK_SIZE: usize = 128;
/// Let 20% of GPU memory be free, this is an arbitrary value.
const MEMORY_PADDING: f64 = 0.2f64;
/// The Nvidia Ampere architecture is compute capability major version 8.
const AMPERE: u32 = 8;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// The number of units the work is split into. One unit will result in one CUDA thread.
///
/// Based on empirical results, it turns out that on Nvidia devices with the Ampere architecture,
/// it's faster to use two times the number of work units.
pub const fn work_units(compute_units: u32, compute_capabilities: Option<(u32, u32)>) -> usize {
    match compute_capabilities {
        Some((AMPERE, _)) => LOCAL_WORK_SIZE * compute_units as usize * 2,
        _ => LOCAL_WORK_SIZE * compute_units as usize,
    }
}

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    program: Program,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    n: usize,
    /// The number of units the work is split into. It will results in this amount of threads on
    /// the GPU.
    work_units: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G::Scalar>,
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
fn calc_chunk_size<G>(mem: u64, work_units: usize) -> usize
where
    G: PrimeCurveAffine,
    G::Scalar: PrimeField,
{
    let aff_size = std::mem::size_of::<G>();
    let exp_size = exp_size::<G::Scalar>();
    let proj_size = std::mem::size_of::<G::Curve>();

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;
    // The amount of memory (in bytes) of a single term.
    let term_size = aff_size + exp_size;
    // The number of buckets needed for one work unit
    let max_buckets_per_work_unit = 1 << MAX_WINDOW_SIZE;
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = work_units * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = work_units * proj_size;

    (max_memory - buckets_size - results_size) / term_size
}

/// The size of the exponent in bytes.
///
/// It's the actual bytes size it needs in memory, not it's theoratical bit size.
fn exp_size<F: PrimeField>() -> usize {
    std::mem::size_of::<F>()
}

impl<'a, G> SingleMultiexpKernel<'a, G>
where
    G: PrimeCurveAffine + GpuName,
{
    /// Create a new Multiexp kernel instance for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        program: Program,
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let mem = device.memory();
        let compute_units = device.compute_units();
        let compute_capability = device.compute_capability();
        let work_units = work_units(compute_units, compute_capability);
        let chunk_size = calc_chunk_size::<G>(mem, work_units);

        Ok(SingleMultiexpKernel {
            program,
            n: chunk_size,
            work_units,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp(&self, bases: &[G], exponents: &[G::Scalar]) -> EcResult<G::Curve> {
        self.multiexp_bound(bases, exponents, 256usize)
    }

    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp_bound(
        &self,
        bases: &[G],
        exponents: &[G::Scalar],
        bits: usize,
    ) -> EcResult<G::Curve> {
        assert_eq!(bases.len(), exponents.len());

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }
        let window_size = self.calc_window_size(bases.len());
        // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.

        let num_windows = div_ceil(bits, window_size);
        let num_groups = self.work_units / num_windows;
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Curve>> {
            let base_buffer = program.create_buffer_from_slice(bases).unwrap();
            let exp_buffer = program.create_buffer_from_slice(exponents).unwrap();

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Curve>(self.work_units * bucket_len).unwrap() };
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Curve>(self.work_units).unwrap() };

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

            let kernel_name = format!("{}_multiexp", G::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE).unwrap();

            kernel
                .arg(&base_buffer)
                .arg(&bucket_buffer)
                .arg(&result_buffer)
                .arg(&exp_buffer)
                .arg(&(bases.len() as u32))
                .arg(&(num_groups as u32))
                .arg(&(num_windows as u32))
                .arg(&(window_size as u32))
                .arg(&(bits as u32))
                .run()?;

            let mut results = vec![G::Curve::identity(); num_windows];
            let acc_buffer = program.create_buffer_from_slice(&results).unwrap();

            let mut max_step = num_groups;

            while max_step > 1 {
                let step = max_step.next_power_of_two() >> 1;
                let global_work_size = div_ceil(num_windows * step, 1);

                let kernel_name = format!("{}_multiexp_group_step", G::name());
                let kernel = program.create_kernel(&kernel_name, global_work_size, 1)?;

                kernel
                    .arg(&result_buffer)
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(step as u32))
                    .run()?;

                max_step = step;
            }

            let kernel_name = format!("{}_multiexp_group_collect", G::name());
            let kernel = program.create_kernel(&kernel_name, num_windows, 1).unwrap();

            kernel
                .arg(&result_buffer)
                .arg(&acc_buffer)
                .arg(&(num_windows as u32))
                .run()?;

            program.read_into_buffer(&acc_buffer, &mut results).unwrap();

            Ok(results)
        });

        let results = self.program.run(closures, ()).unwrap();

        let mut acc = G::Curve::identity();
        let mut acc_bits = 0;

        for i in 0..num_windows {
            let w = std::cmp::min(window_size, bits - acc_bits);
            if i != 0 {
                for _ in 0..w {
                    acc = acc.double();
                }
            }
            acc.add_assign(&results[i]);
            acc_bits += w; // Process the next window
        }

        Ok(acc)
    }

    /// Calculates the window size, based on the given number of terms.
    ///
    /// For best performance, the window size is reduced, so that maximum parallelism is possible.
    /// If you e.g. have put only a subset of the terms into the GPU memory, then a smaller window
    /// size leads to more windows, hence more units to work on, as we split the work into
    /// `num_windows * num_groups`.
    fn calc_window_size(&self, num_terms: usize) -> usize {
        // The window size was determined by running the `gpu_multiexp_consistency` test and
        // looking at the resulting numbers.
        let window_size = ((div_ceil(num_terms, self.work_units) as f64).log2() as usize) + 2;
        std::cmp::min(window_size, MAX_WINDOW_SIZE)
    }

    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp_bound_and_ifft(
        &self,
        bases: &[G],
        exponents: &mut [G::Scalar],
        bits: usize,
        omega: &G::Scalar,
        divisor: &G::Scalar,
        log_n: u32,
    ) -> EcResult<G::Curve>
    where
        G::Scalar: GpuName,
    {
        assert_eq!(bases.len(), exponents.len());

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }
        let window_size = self.calc_window_size(bases.len());
        // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.

        let num_windows = div_ceil(bits, window_size);
        let num_groups = self.work_units / num_windows;
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, arg: &mut _| -> EcResult<Vec<G::Curve>> {
            let exponents = arg;
            let base_buffer = program.create_buffer_from_slice(bases)?;
            let exp_buffer = program.create_buffer_from_slice(exponents)?;

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Curve>(self.work_units * bucket_len)? };
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Curve>(self.work_units)? };

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

            let kernel_name = format!("{}_multiexp", G::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&base_buffer)
                .arg(&bucket_buffer)
                .arg(&result_buffer)
                .arg(&exp_buffer)
                .arg(&(bases.len() as u32))
                .arg(&(num_groups as u32))
                .arg(&(num_windows as u32))
                .arg(&(window_size as u32))
                .arg(&(bits as u32))
                .run()?;

            let mut results = vec![G::Curve::identity(); num_windows];
            {
                let acc_buffer = program.create_buffer_from_slice(&results)?;

                let mut max_step = num_groups;

                while max_step > 1 {
                    let step = max_step.next_power_of_two() >> 1;
                    let global_work_size = div_ceil(num_windows * step, 1);

                    let kernel_name = format!("{}_multiexp_group_step", G::name());
                    let kernel = program.create_kernel(&kernel_name, global_work_size, 1)?;

                    kernel
                        .arg(&result_buffer)
                        .arg(&(num_groups as u32))
                        .arg(&(num_windows as u32))
                        .arg(&(step as u32))
                        .run()?;

                    max_step = step;
                }

                let kernel_name = format!("{}_multiexp_group_collect", G::name());
                let kernel = program.create_kernel(&kernel_name, num_windows, 1)?;

                kernel
                    .arg(&result_buffer)
                    .arg(&acc_buffer)
                    .arg(&(num_windows as u32))
                    .run()?;

                program.read_into_buffer(&acc_buffer, &mut results)?;
            }

            // fft
            {
                use ff::Field;
                use std::ops::MulAssign;

                const MAX_LOG2_RADIX: u32 = 8;
                const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7;
                const LOG2_MAX_ELEMENTS: usize = 32;

                let n = 1 << log_n;
                assert_eq!(exponents.len(), n);
                // All usages are safe as the buffers are initialized from either the host or the GPU
                // before they are read.

                //let timer = start_timer!(|| format!("create buffer"));
                let mut src_buffer = unsafe { program.create_buffer::<G::Scalar>(n)? };
                let mut dst_buffer = unsafe { program.create_buffer::<G::Scalar>(n)? };
                //end_timer!(timer);

                // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
                let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

                // Precalculate:
                // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]

                //let timer = start_timer!(|| "gen omega");
                let mut pq = vec![G::Scalar::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
                pq[0] = G::Scalar::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i].mul_assign(&twiddle);
                    }
                }
                //end_timer!(timer);
                let pq_buffer = program.create_buffer_from_slice(&pq)?;

                // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
                let mut omegas = vec![G::Scalar::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow_vartime([2u64]);
                }
                let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

                //let timer = start_timer!(|| format!("fft copy {}", log_n));
                program.write_from_buffer(&mut src_buffer, &*exponents)?;
                //end_timer!(timer);

                //let timer = start_timer!(|| format!("fft main {}", log_n));
                // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
                let mut log_p = 0u32;
                // Each iteration performs a FFT round
                while log_p < log_n {
                    //let timer = start_timer!(|| format!("round {}", log_p));
                    if let Some(maybe_abort) = &self.maybe_abort {
                        if maybe_abort() {
                            return Err(EcError::Aborted);
                        }
                    }

                    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                    let deg = std::cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << std::cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel_name = format!("{}_radix_fft", G::Scalar::name());
                    let kernel = program.create_kernel(
                        &kernel_name,
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&src_buffer)
                        .arg(&dst_buffer)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&rust_gpu_tools::LocalBuffer::<G::Scalar>::new(1 << deg))
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                    //end_timer!(timer);
                }

                let divisor = vec![*divisor];
                let divisor_buffer = program.create_buffer_from_slice(&divisor[..])?;
                let local_work_size = 128;
                let global_work_size = n / local_work_size;
                let kernel_name = format!("{}_eval_mul_c", G::Scalar::name());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&src_buffer)
                    .arg(&0u32)
                    .arg(&divisor_buffer)
                    .arg(&(n as u32))
                    .run()?;

                program.read_into_buffer(&src_buffer, exponents)?;
            }

            Ok(results)
        });

        let results = self.program.run(closures, exponents)?;

        let mut acc = G::Curve::identity();
        let mut acc_bits = 0;

        for i in 0..num_windows {
            let w = std::cmp::min(window_size, bits - acc_bits);
            if i != 0 {
                for _ in 0..w {
                    acc = acc.double();
                }
            }
            acc.add_assign(&results[i]);
            acc_bits += w; // Process the next window
        }

        Ok(acc)
    }
}

/// A struct that constrains several multiexp kernels for different devices.
pub struct MultiexpKernel<'a, G>
where
    G: PrimeCurveAffine,
{
    /// a set of single multiexp kernels for different devices.
    pub kernels: Vec<SingleMultiexpKernel<'a, G>>,
}

impl<'a, G> MultiexpKernel<'a, G>
where
    G: PrimeCurveAffine + GpuName,
{
    /// Create new kernels, one for each given device.
    pub fn create(programs: Vec<Program>, devices: &[&Device]) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, Some(maybe_abort))
    }

    fn create_optional_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let kernels: Vec<_> = programs
            .into_iter()
            .zip(devices.iter())
            .filter_map(|(program, device)| {
                let device_name = program.device_name().to_string();
                let kernel = SingleMultiexpKernel::create(program, device, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device_name, e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        debug!("Multiexp: {} working device(s) selected.", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            debug!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                i,
                k.program.device_name(),
                k.n
            );
        }
        Ok(MultiexpKernel { kernels })
    }

    /// Calculate multiexp on all available GPUs.
    ///
    /// It needs to run within a [`yastl::Scope`]. This method usually isn't called directly, use
    /// [`MultiexpKernel::multiexp`] instead.
    pub fn parallel_multiexp<'s>(
        &'s mut self,
        scope: &Scope<'s>,
        bases: &'s [G],
        exps: &'s [G::Scalar],
        results: &'s mut [G::Curve],
        error: Arc<RwLock<EcResult<()>>>,
    ) {
        let num_devices = self.kernels.len();
        let num_exps = exps.len();
        // The maximum number of exponentiations per device.
        let chunk_size = ((num_exps as f64) / (num_devices as f64)).ceil() as usize;

        for (((bases, exps), kern), result) in bases
            .chunks(chunk_size)
            .zip(exps.chunks(chunk_size))
            // NOTE vmx 2021-11-17: This doesn't need to be a mutable iterator. But when it isn't
            // there will be errors that the OpenCL CommandQueue cannot be shared between threads
            // safely.
            .zip(self.kernels.iter_mut())
            .zip(results.iter_mut())
        {
            let error = error.clone();
            scope.execute(move || {
                let mut acc = G::Curve::identity();
                for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
                    if error.read().unwrap().is_err() {
                        break;
                    }
                    match kern.multiexp(bases, exps) {
                        Ok(result) => acc.add_assign(&result),
                        Err(e) => {
                            *error.write().unwrap() = Err(e);
                            break;
                        }
                    }
                }
                if error.read().unwrap().is_ok() {
                    *result = acc;
                }
            });
        }
    }

    /// Calculate multiexp.
    ///
    /// This is the main entry point.
    pub fn multiexp(
        &mut self,
        pool: &Worker,
        bases_arc: Arc<Vec<G>>,
        exps: Arc<Vec<G::Scalar>>,
        skip: usize,
    ) -> EcResult<G::Curve> {
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases_arc[skip..(skip + exps.len())];
        let exps = &exps[..];

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        pool.scoped(|s| {
            results = vec![G::Curve::identity(); self.kernels.len()];
            self.parallel_multiexp(s, bases, exps, &mut results, error.clone());
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;

        let mut acc = G::Curve::identity();
        for r in results {
            acc.add_assign(&r);
        }

        Ok(acc)
    }

    /// Returns the number of kernels (one per device).
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }
}
