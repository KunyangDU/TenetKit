cmake_minimum_required(VERSION 3.25)
project(TenetKit
    VERSION     2.0.0
    DESCRIPTION "C++20 tensor-network library for quantum many-body simulations"
    HOMEPAGE_URL "https://github.com/KunyangDU/TenetKit"
    LANGUAGES   CXX
)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # -std=c++20, not -std=gnu++20

# ── Build options ─────────────────────────────────────────────────────────────
option(TENET_SANITIZE "Enable AddressSanitizer + UBSan in Debug builds" OFF)
option(TENET_BUILD_EXAMPLES "Build example programs" ON)

# ── Compiler flags & external dependencies ────────────────────────────────────
include(cmake/CompilerFlags.cmake)
include(cmake/Dependencies.cmake)

# ── Core static library ───────────────────────────────────────────────────────
add_library(tenet STATIC
    # Tensor core
    src/core/dense_tensor.cpp
    src/core/tensor_ops.cpp
    src/core/factorization.cpp

    # MPS / MPO
    src/mps/mps.cpp
    src/mps/mpo.cpp
    src/mps/canonical.cpp

    # Environment
    src/environment/environment.cpp
    src/environment/env_push.cpp

    # Hamiltonian
    src/hamiltonian/projective_ham.cpp
    src/hamiltonian/ham_action.cpp

    # Interaction tree (sparse MPO builder)
    src/intr_tree/interaction_tree.cpp

    # Observables
    src/observables/add_observable.cpp
    src/observables/cal_observable.cpp

    # Algorithms
    src/algorithm/dmrg.cpp
    src/algorithm/tdvp.cpp
    src/algorithm/cbe.cpp
    src/algorithm/settn.cpp

    # Krylov solvers
    src/krylov/lanczos.cpp
    src/krylov/arnoldi.cpp
)

# Public include path (BUILD_INTERFACE is used during build, INSTALL_INTERFACE
# after installation – keeps the tree clean).
target_include_directories(tenet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(tenet PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
    $<$<BOOL:${HDF5_FOUND}>:HDF5::HDF5>
    $<$<BOOL:${USE_MKL}>:MKL::MKL>
)

# Per-config compile options
target_compile_options(tenet PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<AND:$<CONFIG:Debug>,$<BOOL:${TENET_SANITIZE}>>:
        -fsanitize=address,undefined -fno-omit-frame-pointer>
)

target_link_options(tenet PUBLIC
    $<$<AND:$<CONFIG:Debug>,$<BOOL:${TENET_SANITIZE}>>:
        -fsanitize=address,undefined>
)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Examples ──────────────────────────────────────────────────────────────────
if(TENET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Install rules (standard for distributable C++ libraries) ──────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tenet
    EXPORT  TenetKitTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/tenet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate TenetKitConfig.cmake so downstream projects can do:
#   find_package(TenetKit REQUIRED)
#   target_link_libraries(myapp PRIVATE TenetKit::tenet)
install(EXPORT TenetKitTargets
    FILE      TenetKitTargets.cmake
    NAMESPACE TenetKit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TenetKit
)
configure_package_config_file(
    cmake/TenetKitConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TenetKitConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TenetKit
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TenetKitConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TenetKitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TenetKitConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TenetKit
)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== TenetKit v${PROJECT_VERSION} ===")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  OpenMP       : ${OpenMP_FOUND}")
message(STATUS "  HDF5         : ${HDF5_FOUND}")
message(STATUS "  MKL          : ${USE_MKL}")
message(STATUS "  Sanitizers   : ${TENET_SANITIZE}")
message(STATUS "")
