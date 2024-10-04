project (occa-tool VERSION 0.0.1 LANGUAGES CXX)

CPMAddPackage(
  NAME argparser
  SOURCE_DIR ${ROOT_DIR}/deps/argparse
  OPTIONS "ARGPARSE_BUILD_TESTS OFF"
)

CPMAddPackage(NAME spdlog
  SOURCE_DIR ${ROOT_DIR}/deps/spdlog
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
