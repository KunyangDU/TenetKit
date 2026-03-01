# cmake/CompilerFlags.cmake
# Compiler diagnostics and optimization flags for TenetCpp.

# Common warnings for all compilers
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wnon-virtual-dtor
    -Wno-unused-parameter   # Too noisy during early development; re-enable later
)

# Apple Clang needs -Wno-c++20-extensions silenced occasionally
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-Wno-c++20-extensions)
endif()

# Link-Time Optimization in Release builds
# Note: LTO is disabled for AppleClang/clang on macOS by default because
#       it requires lld or the Apple linker to cooperate. Enable explicitly
#       if you are sure your toolchain supports it.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Export compile commands so clangd / clang-tidy can find them
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
