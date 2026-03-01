# cmake/FindMKL.cmake
# Locate Intel oneAPI MKL.
#
# Searches in order:
#  1. MKLConfig.cmake shipped with oneAPI (preferred, sets MKL::MKL target)
#  2. Environment variable MKLROOT
#  3. Common installation paths (/opt/intel/oneapi/mkl, ~/intel)
#
# After inclusion, MKL_FOUND and the imported target MKL::MKL are available.

set(MKL_FOUND FALSE)

# ── Strategy 1: use the CMake config shipped with oneAPI ─────────────────────
find_package(MKL CONFIG QUIET
    HINTS
        $ENV{MKLROOT}/lib/cmake/mkl
        /opt/intel/oneapi/mkl/latest/lib/cmake/mkl
        $ENV{HOME}/intel/oneapi/mkl/latest/lib/cmake/mkl
)
if(MKL_FOUND)
    message(STATUS "FindMKL: found via MKLConfig.cmake")
    return()
endif()

# ── Strategy 2: manual search ────────────────────────────────────────────────
set(_mkl_hints
    $ENV{MKLROOT}
    /opt/intel/oneapi/mkl/latest
    $ENV{HOME}/intel/oneapi/mkl/latest
    /opt/intel/mkl
)

find_path(MKL_INCLUDE_DIR mkl.h
    HINTS ${_mkl_hints}
    PATH_SUFFIXES include
)

find_library(MKL_CORE_LIB    NAMES mkl_core    HINTS ${_mkl_hints} PATH_SUFFIXES lib lib/intel64)
find_library(MKL_ILP64_LIB   NAMES mkl_intel_ilp64  HINTS ${_mkl_hints} PATH_SUFFIXES lib lib/intel64)
find_library(MKL_THREAD_LIB  NAMES mkl_intel_thread  HINTS ${_mkl_hints} PATH_SUFFIXES lib lib/intel64)
find_library(MKL_OMP_LIB     NAMES iomp5 HINTS ${_mkl_hints} PATH_SUFFIXES lib lib/intel64 ../compiler/latest/lib)

if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_ILP64_LIB AND MKL_THREAD_LIB)
    set(MKL_FOUND TRUE)

    if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED)
        target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
        target_compile_definitions(MKL::MKL INTERFACE MKL_ILP64)
        target_link_libraries(MKL::MKL INTERFACE
            -Wl,--start-group
            "${MKL_ILP64_LIB}"
            "${MKL_THREAD_LIB}"
            "${MKL_CORE_LIB}"
            -Wl,--end-group
        )
        if(MKL_OMP_LIB)
            target_link_libraries(MKL::MKL INTERFACE "${MKL_OMP_LIB}")
        endif()
    endif()

    message(STATUS "FindMKL: found via manual search (${MKL_INCLUDE_DIR})")
else()
    message(STATUS "FindMKL: Intel MKL NOT found")
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_CORE_LIB MKL_ILP64_LIB MKL_THREAD_LIB MKL_OMP_LIB)
