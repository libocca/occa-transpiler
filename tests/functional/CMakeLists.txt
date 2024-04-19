CPMAddPackage(
  googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  OPTIONS "INSTALL_GTEST OFF" "INSTALL_GMOCK OFF"
)

CPMAddPackage(
  NAME argparser
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/argparse
  OPTIONS "ARGPARSE_BUILD_TESTS OFF"
)

CPMAddPackage(NAME nlohmann_json
   SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/json
   OPTIONS "JSON_BuildTests OFF")

CPMAddPackage(NAME spdlog
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/spdlog
  OPTIONS "SPDLOG_BUILD_EXAMPLE OFF" "SPDLOG_NO_EXCEPTIONS ON")

add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest)

enable_testing()
include(GoogleTest)

set(TESTS_SOURCES
    common/load_test_suites.cpp
    common/load_test_suites.h
    common/data_directory.h
    common/data_directory.cpp
    generic_configurable_tests.cpp
    internal/test_kernel_info.cpp
    main.cpp
)

add_executable(occa-transpiler-tests ${TESTS_SOURCES})

target_include_directories(occa-transpiler-tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )

target_compile_definitions(occa-transpiler-tests
	PRIVATE
        ${LLVM_DEFINITIONS})

target_link_libraries(occa-transpiler-tests
        PRIVATE
        GTest::GTest
        argparse
	nlohmann_json::nlohmann_json
        occa-transpiler
        spdlog::spdlog_header_only
)

add_custom_command(TARGET occa-transpiler-tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/configs ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/configs
)

gtest_discover_tests(
        occa-transpiler-tests
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        EXTRA_ARGS -s ./configs/test_suite_normalize -d data
        TEST_SUFFIX .noArgs
        TEST_LIST noArgsTests
)
