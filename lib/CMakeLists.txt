cmake_minimum_required(VERSION 3.26)
project (occa-transpiler VERSION 0.0.1 LANGUAGES CXX)

set (OCCA_TRANSPILER_SOURCES
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/ast_visitor.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_ast_consumer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/ast_traversal/transpile_frontend_action.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/diag/diag_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/diag/diag_consumer.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attr_decl_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attr_stmt_handler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/attribute_manager.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/backend_attribute_map.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_manager/common_attribute_map.h

    ${CMAKE_SOURCE_DIR}/include/oklt/core/transpiler_session/transpiler_session.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/utils/format.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/attribute_names.h
    ${CMAKE_SOURCE_DIR}/include/oklt/core/config.h

    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/transpiler/transpiler.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/stages/normalizer/normalizer.h
    ${CMAKE_SOURCE_DIR}/include/oklt/pipeline/normalize_and_transpile.h

    ${CMAKE_SOURCE_DIR}/include/oklt/util/io_helper.h
    ${CMAKE_SOURCE_DIR}/include/oklt/util/string_utils.h

    attributes/frontend/barrier.cpp
    attributes/frontend/dim.cpp
    attributes/frontend/inner.cpp
    attributes/frontend/kernel.cpp
    attributes/frontend/outer.cpp
    attributes/frontend/restrict.cpp
    attributes/frontend/shared.cpp
    attributes/frontend/tile.cpp

    attributes/backend/cuda/kernel.cpp
    attributes/backend/cuda/shared.cpp
    attributes/backend/common/dim.cpp

    core/ast_traversal/ast_visitor.cpp
    core/ast_traversal/transpile_ast_consumer.cpp
    core/ast_traversal/transpile_frontend_action.cpp

    core/diag/diag_consumer.cpp

    core/attribute_manager/attr_decl_handler.cpp
    core/attribute_manager/attr_stmt_handler.cpp
    core/attribute_manager/attribute_manager.cpp
    core/attribute_manager/backend_attribute_map.cpp
    core/attribute_manager/common_attribute_map.cpp

    core/transpiler_session/transpiler_session.cpp
    core/config.cpp

    core/utils/format.cpp

    util/io_helper.cpp
    util/string_utils.cpp

    pipeline/stages/normalize_and_transpile.cpp
    pipeline/stages/transpiler/transpiler.cpp
    pipeline/stages/normalizer/normalizer.cpp
    pipeline/stages/normalizer/impl/okl_to_gnu_stage.cpp
    pipeline/stages/normalizer/impl/gnu_to_std_cpp_stage.cpp
    pipeline/stages/normalizer/impl/okl_attr_traverser.cpp
)

CPMAddPackage(expected
    GITHUB_REPOSITORY
    TartanLlama/expected
    GIT_TAG v1.1.0)

set(LLVM_LINK_COMPONENTS Support)

add_llvm_library(occa-transpiler SHARED
    ${OCCA_TRANSPILER_SOURCES})

# target_compile_definitions(occa-transpile-lib PUBLIC ${LLVM_DEFINITIONS})
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
    PRIVATE
        clangAST
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
        clangToolingRefactoring
        clangASTMatchers
        clangTransformer
  )
