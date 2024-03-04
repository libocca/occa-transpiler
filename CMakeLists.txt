cmake_minimum_required(VERSION 3.26)

project(occa-transpiler VERSION 0.0.1 LANGUAGES C CXX)

option(TRANSPILER_TESTS "build transpiler library tests" ON)

# Add custom modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH "$ENV{CLANG_INSTALLED_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic" )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

include(cmake/CPM.cmake)

find_package(LLVM 17 REQUIRED CONFIG)
find_package(Clang 17 REQUIRED CONFIG)


set(LLVM_ENABLE_RTTI ON)
set(LLVM_REQUIRES_RTTI ON)

include(AddLLVM)

set(CLANG_LIBS libclang clangTooling)

add_subdirectory(lib)
add_subdirectory(tool)

if(TRANSPILER_TESTS)
   add_subdirectory(tests)
endif()


# occa-transpiler

install(
    TARGETS occa-transpiler
    EXPORT oklt-config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    TARGETS occa-tool
    ARCHIVE DESTINATION ${CMAKE_INSTALL_BINDIR}) 

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/oklt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oklt)

install(
    EXPORT oklt-config
    NAMESPACE oklt::
    FILE oklt-targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oklt)

configure_file(cmake/oklt-config.cmake.in oklt-config.cmake @ONLY)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/oklt-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oklt)

