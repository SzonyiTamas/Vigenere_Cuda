# GPU-Accelerated Vigenère Encryption & Key Recovery (CUDA)

A CUDA C++ project that encrypts a large text with the Vigenère cipher on the GPU, then recovers the secret key through a **known-plaintext attack**, implemented both sequentially on the CPU and in parallel on the GPU, with timing for each.

> Built as the semester project for the **Parallel Programming** university course.

---

## Highlights

- **Custom CUDA kernels** for encryption, keystream derivation and key-length search, written directly against the CUDA runtime API.
- **Shared-memory optimization**: each block caches the key (and the reference keystream segment) in `__shared__` memory, so threads don't all read the same bytes from global memory.
- **Parallel tree reduction**: in the key-length search, each block combines its threads' results with a log₂(n) shared-memory reduction (`stride >>= 1` with `__syncthreads()` between steps).
- **Two levels of parallelism**: one **block per candidate key length** (1–64), and within each block the threads split the text between them with a **grid-stride-style loop**.
- **CPU vs. GPU comparison**: the same attack runs as a sequential CPU baseline and as a GPU version, both checked against the true key and timed with **CUDA events**.
- **Device-global symbols** (`__device__` arrays with `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`) for host–device data transfer.

---

## How It Works

The program generates a random 50,000-character plaintext (A–Z) and a random 10-character key, then:

| Step | Where | Kernel / Function | Description |
|------|-------|-------------------|-------------|
| 1. Encrypt | GPU | `KernelEncrypt` | One thread per character: `C[i] = (P[i] + K[i mod L]) mod 26`. The key is cached in shared memory. |
| 2. Derive keystream | GPU | `KernelKeyStream` | One thread per character: `K[i] = (C[i] − P[i]) mod 26`. |
| 3. Find key length | GPU | `KernelTestPeriods` | Block `L` checks whether the keystream repeats with period `L`. Threads scan the text in a strided loop, and a shared-memory OR-reduction combines the results into one valid/invalid flag. |
| 4. Recover key | Host | — | The shortest valid period `L` is the key length, and the first `L` keystream values are the key. |
| Baseline | CPU | `main` | The same keystream derivation and period search, done sequentially for comparison. |

## Build & Run

**Requirements**
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (the project was set up with CUDA 13.1)
- Visual Studio 2022 (MSVC toolset v143) with the CUDA integration

## Author

**Tamás Szőnyi**
