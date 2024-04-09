#include <oklt/util/string_utils.h>

#include "core/attribute_manager/common_attribute_map.h"
#include "core/transpiler_session/session_stage.h"
#include "tl/expected.hpp"

#include <clang/AST/Stmt.h>
#include <system_error>

namespace oklt {
using namespace clang;

bool CommonAttributeMap::registerHandler(std::string name, AttrDeclHandler handler) {
    auto ret = _declHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
    return ret.second;
}

bool CommonAttributeMap::registerHandler(std::string name, AttrStmtHandler handler) {
    auto ret = _stmtHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
    return ret.second;
}

HandleResult CommonAttributeMap::handleAttr(SessionStage& stage,
                                            const clang::Decl& decl,
                                            const clang::Attr& attr,
                                            const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    auto it = _declHandlers.find(name);
    if (it != _declHandlers.end()) {
        return it->second.handle(stage, decl, attr, params);
    }
    return tl::make_unexpected(Error{std::error_code(),
                                     util::fmt("Warning: no handle for attribute {} for node {} \n",
                                               attr.getNormalizedFullName(),
                                               decl.getDeclKindName())
                                         .value()});
}

HandleResult CommonAttributeMap::handleAttr(SessionStage& stage,
                                            const clang::Stmt& stmt,
                                            const clang::Attr& attr,
                                            const std::any* params) {
    std::string name = attr.getNormalizedFullName();
    auto it = _stmtHandlers.find(name);
    if (it != _stmtHandlers.end()) {
        return it->second.handle(stage, stmt, attr, params);
    }
    return tl::make_unexpected(Error{std::error_code(),
                                     util::fmt("Warning: no handle for attribute {} for node {} \n",
                                               attr.getNormalizedFullName(),
                                               stmt.getStmtClassName())
                                         .value()});
}

bool CommonAttributeMap::hasAttrHandler(const std::string& name) {
    auto declIt = _declHandlers.find(name);
    if (declIt != _declHandlers.cend()) {
        return true;
    }
    auto stmtIt = _stmtHandlers.find(name);
    return stmtIt != _stmtHandlers.cend();
}

}  // namespace oklt
