#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/transpilation.h"
#include "core/transpiler_session/session_stage.h"

namespace {
template <typename KeyType,
          typename MapType,
          typename GenKeyType,
          typename GenMapType,
          typename ActionFunc>
oklt::HandleResult runNodeHandler(KeyType key,
                                  MapType& map_,
                                  GenKeyType gen_key,
                                  GenMapType& gen_map,
                                  ActionFunc action) {
    auto it = map_.find(key);
    if (it == map_.end()) {
        auto gen_it = gen_map.find(gen_key);
        if (gen_it == gen_map.end()) {
            return {};
        }
        return action(gen_it->second);
    }
    return action(it->second);
}
}  // namespace

namespace oklt {
using namespace clang;

AstProcessorManager& AstProcessorManager::instance() {
    static AstProcessorManager manager;
    return manager;
}

bool AstProcessorManager::registerGenericHandle(AstProcessorType procType, DeclNodeHandle handle) {
    auto [_, ret] = _genericDeclHandle.try_emplace(procType, std::move(handle));
    return ret;
}

bool AstProcessorManager::registerGenericHandle(AstProcessorType procType, StmtNodeHandle handle) {
    auto [_, ret] = _genericStmtHandle.try_emplace(procType, std::move(handle));
    return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(KeyType key, DeclNodeHandle handler) {
    auto [_, ret] = _declHandlers.try_emplace(std::move(key), std::move(handler));
    return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(KeyType key, StmtNodeHandle handler) {
    auto [_, ret] = _stmtHandlers.try_emplace(std::move(key), std::move(handler));
    return ret;
}

HandleResult AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                         const Decl& decl,
                                                         SessionStage& stage) {
    return runNodeHandler(std::make_tuple(procType, decl.getKind()),
                          _declHandlers,
                          procType,
                          _genericDeclHandle,
                          [&decl, &stage](auto h) { return h.preAction(decl, stage); });
}

HandleResult AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                          const Decl& decl,
                                                          SessionStage& stage) {
    return runNodeHandler(std::make_tuple(procType, decl.getKind()),
                          _declHandlers,
                          procType,
                          _genericDeclHandle,
                          [&decl, &stage](auto h) { return h.postAction(decl, stage); });
}

HandleResult AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                         const Stmt& stmt,
                                                         SessionStage& stage) {
    return runNodeHandler(std::make_tuple(procType, stmt.getStmtClass()),
                          _stmtHandlers,
                          procType,
                          _genericStmtHandle,
                          [&stmt, &stage](auto h) { return h.preAction(stmt, stage); });
}

HandleResult AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                          const Stmt& stmt,
                                                          SessionStage& stage) {
    return runNodeHandler(std::make_tuple(procType, stmt.getStmtClass()),
                          _stmtHandlers,
                          procType,
                          _genericStmtHandle,
                          [&stmt, &stage](auto h) { return h.postAction(stmt, stage); });
}
}  // namespace oklt
