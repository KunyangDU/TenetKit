# examples/CMakeLists.txt

function(tenet_add_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE tenet)
endfunction()

tenet_add_example(heisenberg_dmrg         heisenberg_dmrg.cpp)
tenet_add_example(hubbard_tdvp            hubbard_tdvp.cpp)
tenet_add_example(finite_temperature_settn finite_temperature_settn.cpp)
