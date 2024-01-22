CPMAddPackage(
  googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
)

CPMAddPackage(
  argparser
  GITHUB_REPOSITORY p-ranav/argparse
  GIT_TAG v3.0
)

CPMAddPackage(NAME nlohmann_json
    VERSION 3.11.2
    GITHUB_REPOSITORY nlohmann/json OPTIONS "JSON_BuildTests OFF")

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
    main.cpp
)

add_executable(occa-transpiler-tests ${TESTS_SOURCES})

target_include_directories(occa-transpiler-tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )

if (NOT STATIC_LIBCLANG)
    llvm_config(occa-transpiler-tests USE_SHARED support)
endif()

target_compile_definitions(occa-transpiler-tests
       	PRIVATE
        ${LLVM_DEFINITIONS})

target_link_libraries(occa-transpiler-tests
        PRIVATE
        GTest::GTest
        argparse
	nlohmann_json::nlohmann_json
        occa-transpiler
        ${CLANG_LIBS}
)

gtest_discover_tests(
        occa-transpiler-tests
        TEST_SUFFIX .noArgs
        TEST_LIST noArgsTests
)