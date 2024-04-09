#pragma once

#include <oklt/core/target_backends.h>

#include "core/attribute_manager/implicit_handlers/decl_handler.h"
#include "core/attribute_manager/implicit_handlers/stmt_handler.h"
#include "core/attribute_manager/result.h"

#include <any>
#include <map>
#include <tl/expected.hpp>
#include <tuple>

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

    bool hasHandler(KeyType key) { return _declHandlers.count(key) || _stmtHandlers.count(key); }

    HandleResult operator()(SessionStage& stage, const clang::Decl& decl);
    HandleResult operator()(SessionStage& stage, const clang::Stmt& stmt);

   private:
    DeclHandlers _declHandlers;
    StmtHandlers _stmtHandlers;
};
}  // namespace oklt
