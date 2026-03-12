set(SCF_BENCH_TARGET psi4_scf_benchmark)

add_executable(
  ${SCF_BENCH_TARGET}
  ../scf/scf_benchmark.cc
  ${CMAKE_CURRENT_LIST_DIR}/../../psi4/psi4/src/read_options.cc
)


target_compile_features(${SCF_BENCH_TARGET} PRIVATE cxx_std_17)

get_filename_component(_bench_cases "${CMAKE_CURRENT_LIST_DIR}/../scf/cases" ABSOLUTE)
target_compile_definitions(
  ${SCF_BENCH_TARGET}
  PRIVATE
    psi4_SHGSHELL_ORDERING=${psi4_SHGSHELL_ORDERING}
    BENCHMARK_CASES_DIR="${_bench_cases}"
)

target_include_directories(
  ${SCF_BENCH_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/..
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${SCF_BENCH_TARGET} PRIVATE -fno-omit-frame-pointer)
endif()

# Use the same library list and linking strategy as the Psi4 core module.
# PRE_LIBRARY_OPTION/POST_LIBRARY_OPTION (--whole-archive on Linux) force all
# static lib objects to be included, avoiding order/circular dependency issues.
get_property(psi4_libmodules GLOBAL PROPERTY LINKLIBLIST)
get_property(psi4_binmodules GLOBAL PROPERTY LINKBINLIST)

target_link_libraries(
  ${SCF_BENCH_TARGET}
  PRIVATE
    ${psi4_binmodules}
    ${PRE_LIBRARY_OPTION}
    ${psi4_libmodules}
    ${POST_LIBRARY_OPTION}
    Python::Python
    pybind11::headers
    tgt::lapack
    Libint2::cxx
    Libint2::int2
)

set_target_properties(
  ${SCF_BENCH_TARGET}
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark/scf
)
