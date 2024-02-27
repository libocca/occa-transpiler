#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool ImplicitHandlerMap::registerHandler(KeyType key, DeclHandler handler) {
    auto ret = _declHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

bool ImplicitHandlerMap::registerHandler(KeyType key, StmtHandler handler) {
    auto ret = _stmtHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

HandleResult ImplicitHandlerMap::operator()(const Decl& decl, SessionStage& stage) {
    auto backend = stage.getBackend();
    auto it = _declHandlers.find(std::make_tuple(backend, decl.getKind()));

    // INFO: implcit handler means that only some specific stmt/decl has specific handler
    //       missing of handler is ok
    if (it == _declHandlers.end()) {
        return true;
    }

    return it->second(decl, stage);
}

HandleResult ImplicitHandlerMap::operator()(const Stmt& stmt, SessionStage& stage) {
    auto backend = stage.getBackend();
    auto it = _stmtHandlers.find(std::make_tuple(backend, stmt.getStmtClass()));

    // INFO: implcit handler means that only some specific stmt/decl has specific handler
    //       missing of handler is ok
    if (it == _stmtHandlers.end()) {
        return true;
    }

    return it->second(stmt, stage);
}

}  // namespace oklt
