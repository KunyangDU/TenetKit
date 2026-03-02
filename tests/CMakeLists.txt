# tests/CMakeLists.txt

if(NOT GTest_FOUND)
    message(STATUS "GTest not found – skipping tests.")
    return()
endif()

include(GoogleTest)

# Helper to register a test executable quickly.
function(tenet_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE tenet GTest::gtest_main)
    gtest_discover_tests(${target})
endfunction()

# ── Unit tests (mirror the module structure) ──────────────────────────────────
tenet_add_test(test_dense_tensor   test_dense_tensor.cpp)
tenet_add_test(test_factorization  test_factorization.cpp)
tenet_add_test(test_mps            test_mps.cpp)
tenet_add_test(test_mpo            test_mpo.cpp)
tenet_add_test(test_environment    test_environment.cpp)
tenet_add_test(test_lanczos        test_lanczos.cpp)

# ── Integration / algorithm tests ─────────────────────────────────────────────
tenet_add_test(test_dmrg           test_dmrg.cpp)
tenet_add_test(test_tdvp           test_tdvp.cpp)
tenet_add_test(test_observables    test_observables.cpp)

# ── New: local-space and utility tests ────────────────────────────────────────
tenet_add_test(test_local_space    test_local_space.cpp)
tenet_add_test(test_utils          test_utils.cpp)

# ── SETTN tests (free_fermion_ref.hpp 在 tests/ 目录下，需额外 include 路径) ────
add_executable(test_settn test_settn.cpp)
target_link_libraries(test_settn PRIVATE tenet GTest::gtest_main)
target_include_directories(test_settn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(test_settn)
