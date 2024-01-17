cmake_minimum_required(VERSION 3.26)
project (occa-tool VERSION 0.0.1 LANGUAGES CXX)

CPMAddPackage(
  argparser
  GITHUB_REPOSITORY p-ranav/argparse
  GIT_TAG v3.0
)

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
        clangAST
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
        clangToolingRefactoring
        clangASTMatchers
        clangTransformer
)

