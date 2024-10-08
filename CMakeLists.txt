cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(occa-transpiler VERSION 0.01 LANGUAGES C CXX)

option(TRANSPILER_TESTS "build transpiler library tests" ON)
option(BUILD_DOCUMENTATION "Create the HTML+PDF API documentation (requires Doxygen)" OFF)

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

set(ROOT_DIR ${PROJECT_SOURCE_DIR})

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

include(cmake/CPM_0.38.6.cmake)
include(cmake/EmbedResource.cmake)
include(cmake/GitSubmodules.cmake)

# So far exactly version 17 is required
# This is due to the fact that compatibility layer is required in transpiler code to support newer versions
# This compatibility layer is topic for future releases
if (NOT OCCA_LOCAL_CLANG_PATH)
    find_package(Clang 17 REQUIRED)
else()
    find_package(Clang 17 PATHS ${OCCA_LOCAL_CLANG_PATH} NO_DEFAULT_PATH)
    find_package(LLVM 17 PATHS ${OCCA_LOCAL_CLANG_PATH} NO_DEFAULT_PATH)
endif()

set(LLVM_ENABLE_RTTI ON)
set(LLVM_REQUIRES_RTTI ON)

include(AddLLVM)

set(CLANG_LIBS libclang clangTooling clangFrontendTool)


init_submodules(${ROOT_DIR})

add_subdirectory(lib)
add_subdirectory(tool)

if(TRANSPILER_TESTS)
   add_subdirectory(tests)
endif()


install(
    TARGETS occa-transpiler
    EXPORT oklt-config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    TARGETS occa-tool
    ARCHIVE DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/oklt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    EXPORT oklt-config
    NAMESPACE occa::
    FILE oklt-targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oklt)

configure_file(cmake/oklt-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/oklt-config.cmake @ONLY)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/oklt-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oklt)

find_package(Doxygen)

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR BUILD_DOCUMENTATION)
    if (DOXYGEN_FOUND)
	set(DOXYGEN_IN ${ROOT_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        add_custom_target(generate_documentation ALL
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                VERBATIM )
    else (DOXYGEN_FOUND)
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif (DOXYGEN_FOUND)
endif()
