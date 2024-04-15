cmake_minimum_required(VERSION 3.26)
project (occa-tool VERSION 0.0.1 LANGUAGES CXX)

CPMAddPackage(
  NAME argparser
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/submodules/argparse
#  GITHUB_REPOSITORY p-ranav/argparse
#  GIT_TAG v3.0
  OPTIONS "ARGPARSE_BUILD_TESTS OFF"
)

CPMAddPackage(NAME spdlog
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/submodules/spdlog
#  VERSION 1.13.0
#  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_BUILD_EXAMPLE OFF" "SPDLOG_NO_EXCEPTIONS ON")


add_executable(occa-tool
        main.cpp
)

target_include_directories(occa-tool
        PRIVATE
        occa-transpiler
)

target_link_libraries(occa-tool
        PRIVATE
        occa-transpiler
        argparse
        spdlog::spdlog_header_only
)

# Make sure that spdlog macro supports all logging levels
target_compile_definitions(occa-tool PRIVATE -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
