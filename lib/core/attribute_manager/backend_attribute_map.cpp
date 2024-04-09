#include <oklt/core/error.h>

#include "core/attribute_manager/backend_attribute_map.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool BackendAttributeMap::registerHandler(KeyType key, AttrDeclHandler handler) {
    auto ret = _declHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

bool BackendAttributeMap::registerHandler(KeyType key, AttrStmtHandler handler) {
    auto ret = _stmtHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

HandleResult BackendAttributeMap::handleAttr(SessionStage& stage,
                                             const clang::Decl& decl,
                                             const clang::Attr& attr,
                                             const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    auto backend = stage.getBackend();
    auto it = _declHandlers.find(std::make_tuple(backend, name));
    if (it == _declHandlers.end()) {
        return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    }
    return it->second.handle(stage, decl, attr, params);
}

HandleResult BackendAttributeMap::handleAttr(SessionStage& stage,
                                             const clang::Stmt& stmt,
                                             const clang::Attr& attr,
                                             const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    auto backend = stage.getBackend();
    auto it = _stmtHandlers.find(std::make_tuple(backend, name));
    if (it == _stmtHandlers.end()) {
        return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    }
    return it->second.handle(stage, stmt, attr, params);
}

bool BackendAttributeMap::hasAttrHandler(SessionStage& stage, const std::string& name) {
    auto key = std::make_tuple(stage.getBackend(), name);
    auto declIt = _declHandlers.find(key);
    if (declIt != _declHandlers.cend()) {
        return true;
    }
    auto stmtIt = _stmtHandlers.find(key);
    return stmtIt != _stmtHandlers.cend();
}
}  // namespace oklt
