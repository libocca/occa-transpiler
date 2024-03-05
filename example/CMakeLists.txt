cmake_minimum_required(VERSION 3.26)
project (occa-example VERSION 0.0.1 LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "$ENV{OKL_TRANSPILER}")

find_package(oklt REQUIRED)

add_executable(occa-example
        main.cpp
)

target_link_libraries(occa-example
        PRIVATE
        occa::occa-transpiler
)

