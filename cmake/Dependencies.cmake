# cmake/Dependencies.cmake
# Locate all external dependencies for TenetCpp.

# ── Eigen3 (required, header-only) ───────────────────────────────────────────
# Version omitted: Homebrew ships Eigen 5.x which uses a different major version
# than the 3.4 we designed against, but the API we use is fully compatible.
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${Eigen3_VERSION}")

# ── OpenMP (optional but strongly recommended) ────────────────────────────────
# On macOS with Apple Clang, libomp must be installed separately:
#   brew install libomp
# then pass -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_LIB_NAMES="omp"
#             -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP NOT found – parallel environment updates disabled.")
endif()

# ── HDF5 (optional for I/O, CXX component) ───────────────────────────────────
find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
else()
    message(WARNING "HDF5 NOT found – MPS save/load will be unavailable.")
endif()

# ── spdlog (required for logging) ────────────────────────────────────────────
find_package(spdlog REQUIRED)
message(STATUS "spdlog found: ${spdlog_VERSION}")

# ── fmt (used by spdlog; pulled in automatically, but state it explicitly) ───
# spdlog's cmake target already brings in fmt; no separate find needed.

# ── Google Test (required for tests) ─────────────────────────────────────────
find_package(GTest)
if(NOT GTest_FOUND)
    message(WARNING "GTest NOT found – unit tests will be skipped.")
endif()

# ── Intel MKL (optional BLAS/LAPACK backend) ─────────────────────────────────
option(USE_MKL "Use Intel MKL as the BLAS/LAPACK backend" ON)
if(USE_MKL)
    include(${CMAKE_CURRENT_LIST_DIR}/FindMKL.cmake)
    if(MKL_FOUND)
        add_compile_definitions(EIGEN_USE_MKL_ALL)
        message(STATUS "Intel MKL found – EIGEN_USE_MKL_ALL enabled.")
    else()
        message(WARNING "USE_MKL=ON but MKL NOT found; falling back to system BLAS.")
        set(USE_MKL OFF CACHE BOOL "" FORCE)
    endif()
endif()
