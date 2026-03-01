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
