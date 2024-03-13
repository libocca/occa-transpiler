cmake_minimum_required(VERSION 3.26)
project (occa-transpiler VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NORMALIZER_DEBUG_LOG "Log normalizer debug messages in stdout" OFF)
option(OKL_SEMA_DEBUG_LOG "Log normalizer debug messages in stdout" OFF)
option(TRANSPILER_DEBUG_LOG "Log transpiler debug messages in stdout" OFF)

set (OCCA_TRANSPILER_SOURCES
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_processor_types.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/target_backends.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/user_input.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/user_output.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/kernel_metadata.h

    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/normalizer_and_transpiler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/normalizer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/transpiler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/util/format.h

    attributes/frontend/barrier.cpp
    attributes/frontend/nobarrier.cpp
    attributes/frontend/exclusive.cpp
    attributes/frontend/atomic.cpp
    attributes/frontend/dim.cpp
    attributes/frontend/dimOrder.cpp
    attributes/frontend/inner.cpp
    attributes/frontend/kernel.cpp
    attributes/frontend/outer.cpp
    attributes/frontend/restrict.cpp
    attributes/frontend/shared.cpp
    attributes/frontend/tile.cpp

    # Backends common
    attributes/utils/parser.h
    attributes/utils/parser.cpp
    attributes/utils/parser_impl.hpp
    attributes/utils/code_gen.h
    attributes/utils/code_gen.cpp
    attributes/utils/replace_attribute.h
    attributes/utils/replace_attribute.cpp
    attributes/utils/tile_utils.cpp
    attributes/utils/tile_utils.h
    attributes/utils/inner_outer_utils.cpp
    attributes/utils/inner_outer_utils.h

    # Cuda subset
    attributes/utils/cuda_subset/kernel.cpp
    attributes/utils/cuda_subset/restrict.cpp
    attributes/utils/cuda_subset/shared.cpp
    attributes/utils/cuda_subset/atomic.cpp
    attributes/utils/cuda_subset/exclusive.cpp
    attributes/utils/cuda_subset/tile.cpp
    attributes/utils/cuda_subset/inner.cpp
    attributes/utils/cuda_subset/outer.cpp
    attributes/utils/cuda_subset/barrier.cpp
    attributes/utils/cuda_subset/handle.h
    attributes/utils/cuda_subset/loop_code_gen.h
    attributes/utils/cuda_subset/loop_code_gen.cpp

    # HIP
    attributes/backend/hip/global_constant.cpp
    attributes/backend/hip/translation_unit.cpp
    attributes/backend/hip/global_function.cpp
    attributes/backend/hip/tile.cpp
    attributes/backend/hip/inner.cpp
    attributes/backend/hip/outer.cpp
    attributes/backend/hip/kernel.cpp
    attributes/backend/hip/atomic.cpp
    attributes/backend/hip/restrict.cpp
    attributes/backend/hip/shared.cpp
    attributes/backend/hip/exclusive.cpp
    attributes/backend/hip/barrier.cpp

    # CUDA
    attributes/backend/cuda/global_constant.cpp
    attributes/backend/cuda/translation_unit.cpp
    attributes/backend/cuda/global_function.cpp
    attributes/backend/cuda/atomic.cpp
    attributes/backend/cuda/kernel.cpp
    attributes/backend/cuda/inner.cpp
    attributes/backend/cuda/outer.cpp
    attributes/backend/cuda/tile.cpp
    attributes/backend/cuda/restrict.cpp
    attributes/backend/cuda/shared.cpp
    attributes/backend/cuda/barrier.cpp
    attributes/backend/cuda/exclusive.cpp

    # DPCPP
    attributes/backend/dpcpp/kernel.cpp
    attributes/backend/dpcpp/translation_unit.cpp
    attributes/backend/dpcpp/global_function.cpp
    attributes/backend/dpcpp/outer.cpp
    attributes/backend/dpcpp/inner.cpp
    attributes/backend/dpcpp/tile.cpp
    attributes/backend/dpcpp/shared.cpp
    attributes/backend/dpcpp/restrict.cpp
    attributes/backend/dpcpp/atomic.cpp
    attributes/backend/dpcpp/barrier.cpp
    attributes/backend/dpcpp/exclusive.cpp
    attributes/backend/dpcpp/common.cpp
    attributes/backend/dpcpp/common.h

    # Serial subset
    attributes/utils/serial_subset/empty.cpp
    attributes/utils/serial_subset/kernel.cpp
    attributes/utils/serial_subset/restrict.cpp
    attributes/utils/serial_subset/shared.cpp
    attributes/utils/serial_subset/exclusive.cpp
    attributes/utils/serial_subset/tile.cpp
    attributes/utils/serial_subset/inner.cpp
    attributes/utils/serial_subset/outer.cpp
    attributes/utils/serial_subset/handle.h

    # Serial
    attributes/backend/serial/kernel.cpp
    attributes/backend/serial/outer.cpp
    attributes/backend/serial/inner.cpp
    attributes/backend/serial/tile.cpp
    attributes/backend/serial/restrict.cpp
    attributes/backend/serial/shared.cpp
    attributes/backend/serial/exclusive.cpp
    attributes/backend/serial/atomic.cpp
    attributes/backend/serial/barrier.cpp

    # OPENMP
    attributes/backend/openmp/kernel.cpp
    attributes/backend/openmp/outer.cpp
    attributes/backend/openmp/inner.cpp
    attributes/backend/openmp/tile.cpp
    attributes/backend/openmp/restrict.cpp
    attributes/backend/openmp/shared.cpp
    attributes/backend/openmp/exclusive.cpp
    attributes/backend/openmp/atomic.cpp
    attributes/backend/openmp/barrier.cpp

    # Launcher
    attributes/backend/launcher.cpp

    # sugar attribute
    attributes/backend/common/dim.cpp
    # attributes/backend/common/dim_order.cpp

    core/ast_processor_manager/ast_processor_manager.cpp

    core/ast_processors/okl_sema_processor/processor.cpp
    core/ast_processors/okl_sema_processor/handlers/function.h
    core/ast_processors/okl_sema_processor/handlers/function.cpp
    core/ast_processors/okl_sema_processor/handlers/loop.h
    core/ast_processors/okl_sema_processor/handlers/loop.cpp

    core/ast_traversal/transpile_ast_consumer.cpp
    core/ast_traversal/transpile_frontend_action.cpp
    core/ast_traversal/preorder_traversal_nlr.cpp

    core/sema/okl_sema_ctx.cpp
    core/sema/okl_sema_ctx.h
    core/sema/okl_sema_info.cpp
    core/sema/okl_sema_info.h

    core/diag/diag_consumer.cpp

    core/kernel_metadata.cpp

    core/vfs/overlay_fs.cpp

    core/attribute_manager/attr_decl_handler.cpp
    core/attribute_manager/attr_stmt_handler.cpp
    core/attribute_manager/attribute_manager.cpp
    core/attribute_manager/attributed_type_map.cpp
    core/attribute_manager/backend_attribute_map.cpp
    core/attribute_manager/common_attribute_map.cpp
    core/attribute_manager/implicit_handlers/implicit_handler_map.cpp
    core/attribute_manager/implicit_handlers/decl_handler.cpp
    core/attribute_manager/implicit_handlers/stmt_handler.cpp

    core/transpiler_session/transpiler_session.cpp
    core/transpiler_session/session_stage.cpp
    core/transpiler_session/header_info.cpp
    core/transpiler_session/kernel_generator.cpp
    core/transpiler_session/kernel_metadata_generator.h
    core/transpiler_session/kernel_metadata_generator.cpp

    core/target_backends.cpp

    core/utils/format.cpp
    core/utils/type_converter.cpp
    core/utils/for_stmt_parser.cpp
    core/utils/attributes.cpp
    core/utils/attributes.h
    core/utils/var_decl.cpp
    core/utils/var_decl.h
    core/utils/range_to_string.h
    core/utils/range_to_string.cpp

    util/io_helper.cpp
    util/string_utils.cpp

    pipeline/normalizer.cpp
    pipeline/transpiler.cpp
    pipeline/normalizer_and_transpiler.cpp
    pipeline/stages/transpiler/transpiler.cpp
    pipeline/stages/transpiler/transpiler.h
    pipeline/stages/transpiler/error_codes.cpp
    pipeline/stages/transpiler/error_codes.h

    pipeline/stages/normalizer/normalizer.cpp
    pipeline/stages/normalizer/normalizer.h
    pipeline/stages/normalizer/error_codes.cpp
    pipeline/stages/normalizer/error_codes.h
    pipeline/stages/normalizer/impl/expand_macro_stage.cpp
    pipeline/stages/normalizer/impl/okl_to_gnu_stage.cpp
    pipeline/stages/normalizer/impl/gnu_to_std_cpp_stage.cpp
    pipeline/stages/normalizer/impl/okl_attr_traverser.cpp
)

CPMAddPackage(expected
    GITHUB_REPOSITORY
    TartanLlama/expected
    GIT_TAG v1.1.0
    OPTIONS "EXPECTED_BUILD_TESTS OFF" "EXPECTED_BUILD_PACKAGE OFF"
)

CPMAddPackage(NAME nlohmann_json
    VERSION 3.11.2
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS "JSON_BuildTests OFF")

set(LLVM_LINK_COMPONENTS Support)

add_llvm_library(occa-transpiler SHARED
     ${OCCA_TRANSPILER_SOURCES})

if (NORMALIZER_DEBUG_LOG)
  target_compile_definitions(occa-transpiler PRIVATE -DNORMALIZER_DEBUG_LOG=1)
endif()

if (OKL_SEMA_DEBUG_LOG)
  target_compile_definitions(occa-transpiler PRIVATE -DOKL_SEMA_DEBUG_LOG=1)
endif()

if (TRANSPILER_DEBUG_LOG)
  target_compile_definitions(occa-transpiler PRIVATE -DTRANSPILER_DEBUG_LOG=1)
endif()

target_include_directories(occa-transpiler
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${LIBCLANG_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/lib/
)

target_link_libraries(occa-transpiler
    PUBLIC
        $<BUILD_INTERFACE:tl::expected>
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
        ${CLANG_LIBS}
)

add_custom_command(TARGET occa-transpiler
    # Run after all other rules within the target have been executed
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}
    COMMENT "update compile_command.json"
    VERBATIM
)
