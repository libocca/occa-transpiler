#include <oklt/util/string_utils.h>

#include "core/attribute_manager/common_attribute_map.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Stmt.h>

#include <system_error>
#include <tl/expected.hpp>

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

HandleResult CommonAttributeMap::handleAttr(const Attr& attr,
                                            const Decl& decl,
                                            const std::any* params,
                                            SessionStage& stage) {
    auto name = getOklAttrFullName(attr);
    auto it = _declHandlers.find(name);
    if (it != _declHandlers.end()) {
        return it->second.handle(attr, decl, params, stage);
    }
    return tl::make_unexpected(Error{std::error_code(),
                                     util::fmt("Warning: no handle for attribute {} for node {} \n", name, decl.getDeclKindName())
                                         .value()});
}

HandleResult CommonAttributeMap::handleAttr(const Attr& attr,
                                            const Stmt& stmt,
                                            const std::any* params,
                                            SessionStage& stage) {
    auto name = getOklAttrFullName(attr);
    auto it = _stmtHandlers.find(name);
    if (it != _stmtHandlers.end()) {
        return it->second.handle(attr, stmt, params, stage);
    }
    return tl::make_unexpected(Error{std::error_code(),
                                     util::fmt("Warning: no handle for attribute {} for node {} \n", name, stmt.getStmtClassName())
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
