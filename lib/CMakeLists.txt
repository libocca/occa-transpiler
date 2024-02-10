cmake_minimum_required(VERSION 3.26)
project (occa-transpiler VERSION 0.0.1 LANGUAGES CXX)

option(NORMALIZER_DEBUG_LOG "Log normalizer debug messages in stdout" OFF)

set (OCCA_TRANSPILER_SOURCES
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_ast_consumer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_frontend_action.h
#    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/semantic_category.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/validate_attributes.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/ast_traversal.hpp

    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/composite_sema.hpp
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/default_sema.hpp
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/mock_processor.hpp
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/kernel_sema.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/params_sema.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/attr_stmt_sema.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/recovery_expr_sema.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/sema/sema.hpp

    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_types/for_stmt_info.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_types/function_info.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_types/param_info.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_types/attribute_info.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/diag/diag_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/diag/diag_consumer.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/transpile_changes.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attr_decl_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attr_stmt_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attribute_manager.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attributed_type_map.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/backend_attribute_map.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/common_attribute_map.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/implicit_handlers/implicit_handler_map.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/implicit_handlers/decl_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/implicit_handlers/stmt_handler.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/transpiler_session.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/session_stage.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/session_result.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/user_input.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/user_output.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/utils/format.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_names.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/target_backends.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/kernel_info/kernel_info.h

    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/transpiler/transpiler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/transpiler/error_codes.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/normalizer/normalizer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/normalizer/error_codes.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/normalizer_and_transpiler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/normalizer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/transpiler.h

    ${CMAKE_SOURCE_DIR}/include/oklt/util/io_helper.h
    ${CMAKE_SOURCE_DIR}/include/oklt/util/string_utils.h

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

    attributes/backend/cuda/kernel.cpp
    attributes/backend/cuda/shared.cpp
    attributes/backend/cuda/global_constant.cpp
    attributes/backend/common/dim.cpp

#    core/ast_traversal/semantic_category.cpp
    core/ast_traversal/transpile_ast_consumer.cpp
    core/ast_traversal/transpile_frontend_action.cpp
    core/ast_traversal/validate_attributes.cpp

    core/ast_traversal/transpile_types/function_info.cpp
    core/ast_traversal/transpile_types/param_info.cpp
    core/ast_traversal/transpile_types/attribute_info.cpp

    core/ast_traversal/sema/kernel_sema.cpp
    core/ast_traversal/sema/param_sema.cpp
    core/ast_traversal/sema/attr_stmt_sema.cpp
    core/ast_traversal/sema/recovery_expr_sema.cpp

    core/diag/diag_consumer.cpp

    core/kernel_info/kernel_info.cpp

    core/attribute_manager/attr_decl_handler.cpp
    core/attribute_manager/attr_stmt_handler.cpp
    core/attribute_manager/attribute_manager.cpp
    core/attribute_manager/attributed_type_map.cpp
    core/attribute_manager/backend_attribute_map.cpp
    core/attribute_manager/common_attribute_map.cpp
    core/attribute_manager/implcit_handlers/implicit_handler_map.cpp
    core/attribute_manager/implcit_handlers/decl_handler.cpp
    core/attribute_manager/implcit_handlers/stmt_handler.cpp

    core/transpiler_session/transpiler_session.cpp
    core/transpiler_session/session_stage.cpp
    core/target_backends.cpp

    core/utils/format.cpp

    util/io_helper.cpp
    util/string_utils.cpp

    pipeline/normalizer.cpp
    pipeline/transpiler.cpp
    pipeline/normalizer_and_transpiler.cpp
    pipeline/stages/transpiler/transpiler.cpp
    pipeline/stages/transpiler/error_codes.cpp

    pipeline/stages/normalizer/normalizer.cpp
    pipeline/stages/normalizer/error_codes.cpp
    pipeline/stages/normalizer/impl/okl_to_gnu_stage.cpp
    pipeline/stages/normalizer/impl/gnu_to_std_cpp_stage.cpp
    pipeline/stages/normalizer/impl/okl_attr_traverser.cpp
)

CPMAddPackage(expected
    GITHUB_REPOSITORY
    TartanLlama/expected
    GIT_TAG v1.1.0
    OPTIONS "EXPECTED_BUILD_TESTS OFF" "EXPECTED_BUILD_PACKAGE OFF" ""
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

target_include_directories(occa-transpiler
        PUBLIC

        ${LLVM_INCLUDE_DIRS}
        ${LIBCLANG_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}

        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(occa-transpiler
    PUBLIC
        tl::expected
        nlohmann_json::nlohmann_json
    PRIVATE
        ${CLANG_LIBS}
  )
