# TenetKit

A high-performance C++20 tensor-network library for quantum many-body simulations,
implementing DMRG, TDVP, CBE, and SETTN algorithms on matrix product states (MPS)
and matrix product operators (MPO).

> **Status:** Phase 1 in progress — scaffold and build system complete, core
> implementations pending.

---

## Features

- **Backend-parameterised design** — all data structures are templated on a `Backend`
  type.  `DenseBackend` (Eigen-based, Phase 1) is a drop-in replacement for a future
  `SymmetricBackend` (block-sparse with quantum numbers, Phase 2).
- **C++20** — concepts for compile-time interface checking, `std::span`, `std::optional`,
  structured-binding, and zero-overhead tag dispatch.
- **RAII throughout** — no raw `new`/`delete`; resources owned via `std::vector` and
  `std::unique_ptr`.
- **Algorithm coverage**
  - DMRG (single-site & two-site sweep)
  - TDVP (real-time & imaginary-time / tanTRG)
  - CBE (Correlated Basis Expansion, adaptive bond growth)
  - SETTN (Series Expansion Tensor Network, finite temperature)

---

## Directory layout

```
TenetKit/
├── CMakeLists.txt              # Root build (CMake ≥ 3.25)
├── cmake/
│   ├── CompilerFlags.cmake
│   ├── Dependencies.cmake
│   └── FindMKL.cmake
│
├── include/tenet/              # Public headers (install target)
│   ├── core/                   # TrivialSpace, DenseTensor, Backend concept
│   ├── mps/                    # MPSTensor, DenseMPS, SparseMPO, canonicalisation
│   ├── environment/            # Environment, env-push, CBEEnvironment
│   ├── hamiltonian/            # SparseProjectiveHamiltonian, apply()
│   ├── intr_tree/              # InteractionTree → SparseMPO compiler
│   ├── observables/            # Observable tree, ⟨O⟩ calculator
│   ├── local_space/            # Built-in spin / fermion operator matrices
│   ├── algorithm/              # dmrg, tdvp, cbe, settn
│   ├── krylov/                 # Lanczos eigensolver, Arnoldi exp(tH)v
│   ├── process_control/        # Config structs, tag types, sweep info
│   └── utils/                  # Logger (spdlog), Timer, HDF5 I/O helpers
│
├── src/                        # Implementation files (compiled into libtenet)
├── tests/                      # Google Test unit & integration tests
├── examples/                   # Heisenberg DMRG, Hubbard TDVP, SETTN
└── legacy/                     # Reference Julia source (read-only)
```

---

## Dependencies

| Library      | Version  | Required | Purpose                              |
|--------------|----------|----------|--------------------------------------|
| **CMake**    | ≥ 3.25   | Yes      | Build system                         |
| **Eigen3**   | ≥ 3.4    | Yes      | Dense matrix algebra (BLAS ZGEMM)    |
| **spdlog**   | ≥ 1.11   | Yes      | Structured logging                   |
| **Google Test** | ≥ 1.13 | Tests   | Unit / integration tests             |
| **OpenMP**   | ≥ 4.5    | No       | Parallel environment updates         |
| **HDF5**     | ≥ 1.12   | No       | MPS save / load                      |
| **Intel MKL**| latest   | No       | High-performance BLAS backend        |

### Install on macOS (Homebrew)

```bash
brew install eigen spdlog googletest
# Optional:
brew install libomp hdf5
```

---

## Building

```bash
# Configure (Debug, no MKL)
cmake -B build \
      -DCMAKE_BUILD_TYPE=Debug \
      -DUSE_MKL=OFF \
      -DCMAKE_PREFIX_PATH=/opt/homebrew

# Build
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure
```

### CMake options

| Option              | Default | Description                                  |
|---------------------|---------|----------------------------------------------|
| `USE_MKL`           | `ON`    | Link Intel MKL as BLAS backend               |
| `TENET_SANITIZE`    | `OFF`   | Enable AddressSanitizer + UBSan (Debug only) |
| `TENET_BUILD_EXAMPLES` | `ON` | Build example executables                    |

### Release build

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build-release -j$(nproc)
```

### With AddressSanitizer

```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DTENET_SANITIZE=ON -DUSE_MKL=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build-asan -j$(nproc)
```

---

## Quick start (after Phase 1 implementation)

```cpp
#include "tenet/algorithm/dmrg.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/local_space/spin.hpp"

int main() {
    using namespace tenet;
    constexpr int L = 20;

    // Build Heisenberg Hamiltonian via interaction tree
    InteractionTree<> tree(L);
    auto phys = TrivialSpace::trivial(2);   // spin-1/2
    for (int i = 0; i < L - 1; ++i) {
        add_intr2(tree,
            make_op(spin::half::Sz(), i),
            make_op(spin::half::Sz(), i + 1), 1.0);
        add_intr2(tree,
            make_op(spin::half::Sp(), i),
            make_op(spin::half::Sm(), i + 1), 0.5);
        add_intr2(tree,
            make_op(spin::half::Sm(), i),
            make_op(spin::half::Sp(), i + 1), 0.5);
    }
    auto H = compile(tree);

    // Random initial MPS, D = 32
    auto psi = MPS::random(L, 32, std::vector(L, phys));

    // Run DMRG
    Environment env(psi, H);
    env.build_all();
    auto result = dmrg1(env, {.max_sweeps = 20, .E_tol = 1e-8});

    return 0;
}
```

---

## Development roadmap

| Phase | Milestone                            | Status      |
|-------|--------------------------------------|-------------|
| 1.1   | DenseTensor, tensor ops, SVD/QR      | Pending     |
| 1.2   | DenseMPS, SparseMPO, canonicalisation | Pending    |
| 1.3   | LocalSpace + InteractionTree → MPO   | Pending     |
| 1.4   | Environment push & build             | Pending     |
| 1.5   | Lanczos eigensolver + H·v action     | Pending     |
| 1.6   | DMRG (single-site & two-site)        | Pending     |
| 1.7   | TDVP, CBE, SETTN                     | Pending     |
| 2.0   | SymmetricBackend (block-sparse, U(1)/SU(2)) | Future |

---

## License

MIT © 2026 KunyangDU
