#pragma once

#include <oklt/core/target_backends.h>

#include "core/attribute_manager/implicit_handlers/decl_handler.h"
#include "core/attribute_manager/implicit_handlers/stmt_handler.h"

#include <map>
#include <tuple>
#include <any>
#include <tl/expected.hpp>

namespace oklt {
class ImplicitHandlerMap {
   public:
    using KeyType = std::tuple<TargetBackend, int>;
    using DeclHandlers = std::map<KeyType, DeclHandler>;
    using StmtHandlers = std::map<KeyType, StmtHandler>;

    ImplicitHandlerMap() = default;
    ~ImplicitHandlerMap() = default;

    bool registerHandler(KeyType key, DeclHandler handler);
    bool registerHandler(KeyType key, StmtHandler handler);

    tl::expected<std::any, Error> operator()(const clang::Decl* decl, SessionStage& stage);
    tl::expected<std::any, Error> operator()(const clang::Stmt* stmt, SessionStage& stage);

   private:
    DeclHandlers _declHandlers;
    StmtHandlers _stmtHandlers;
};
}  // namespace oklt
