#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>

namespace {
using namespace oklt;

template <typename KeyType,
          typename MapType,
          typename GenKeyType,
          typename GenMapType,
          typename ActionFunc>
oklt::HandleResult runNodeHandler(KeyType key,
                                  MapType& attr_map,
                                  GenKeyType gen_key,
                                  GenMapType& gen_map,
                                  ActionFunc action) {
    auto it = attr_map.find(key);
    if (it == attr_map.end()) {
        auto gen_it = gen_map.find(gen_key);
        if (gen_it == gen_map.end()) {
            return {};
        }
        return action(gen_it->second);
    }
    return action(it->second);
}

AstProcessorManager::AttrKeyType makeAttrKey(AstProcessorType procType, const clang::Attr* attr) {
    return {procType, attr ? getOklAttrFullName(*attr) : ""};
}
}  // namespace

namespace oklt {
using namespace clang;

AstProcessorManager& AstProcessorManager::instance() {
    static AstProcessorManager manager;
    return manager;
}

bool AstProcessorManager::registerDefaultHandle(DefaultKeyType key, DeclNodeHandle handle) {
    auto [_, ret] = _defaultDeclHandlers.try_emplace(key, std::move(handle));
    return ret;
}

bool AstProcessorManager::registerDefaultHandle(DefaultKeyType key, StmtNodeHandle handle) {
    auto [_, ret] = _defaultStmtHandlers.try_emplace(key, std::move(handle));
    return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(AttrKeyType key, DeclNodeHandle handler) {
    auto [_, ret] = _declHandlers.try_emplace(std::move(key), std::move(handler));
    return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(AttrKeyType key, StmtNodeHandle handler) {
    auto [_, ret] = _stmtHandlers.try_emplace(std::move(key), std::move(handler));
    return ret;
}

HandleResult AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                         const clang::Attr* attr,
                                                         const Decl& decl,
                                                         OklSemaCtx& sema,
                                                         SessionStage& stage) {
    return runNodeHandler(
        makeAttrKey(procType, attr), _declHandlers, procType, _defaultDeclHandlers, [&](auto h) {
            return h.preAction(attr, decl, sema, stage);
        });
}

HandleResult AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                          const clang::Attr* attr,
                                                          const Decl& decl,
                                                          OklSemaCtx& sema,
                                                          SessionStage& stage) {
    return runNodeHandler(
        makeAttrKey(procType, attr), _declHandlers, procType, _defaultDeclHandlers, [&](auto h) {
            return h.postAction(attr, decl, sema, stage);
        });
}

HandleResult AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                         const clang::Attr* attr,
                                                         const Stmt& stmt,
                                                         OklSemaCtx& sema,
                                                         SessionStage& stage) {
    return runNodeHandler(
        makeAttrKey(procType, attr), _stmtHandlers, procType, _defaultStmtHandlers, [&](auto h) {
            return h.preAction(attr, stmt, sema, stage);
        });
}

HandleResult AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                          const clang::Attr* attr,
                                                          const Stmt& stmt,
                                                          OklSemaCtx& sema,
                                                          SessionStage& stage) {
    return runNodeHandler(
        makeAttrKey(procType, attr), _stmtHandlers, procType, _defaultStmtHandlers, [&](auto h) {
            return h.postAction(attr, stmt, sema, stage);
        });
}
}  // namespace oklt
