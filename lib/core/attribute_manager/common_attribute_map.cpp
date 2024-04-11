#include <oklt/util/string_utils.h>

#include "core/attribute_manager/common_attribute_map.h"
#include "core/transpiler_session/session_stage.h"
#include "tl/expected.hpp"

#include <clang/AST/Stmt.h>
#include <system_error>

namespace oklt {
using namespace clang;

bool CommonAttributeMap::registerHandler(const KeyType key, AttrHandler handler) {
    auto ret = _nodeHandlers.emplace(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

HandleResult CommonAttributeMap::handleAttr(SessionStage& stage,
                                            const clang::DynTypedNode& node,
                                            const clang::Attr& attr,
                                            const std::any* params) {
    auto kind = node.getNodeKind();
    auto name = attr.getNormalizedFullName();
    auto it = _nodeHandlers.find({name, kind});
    if (it != _nodeHandlers.end()) {
        return it->second.handle(stage, node, attr, params);
    }

    it = _nodeHandlers.find({name, kind.getCladeKind()});
    if (it != _nodeHandlers.end()) {
        return it->second.handle(stage, node, attr, params);
    }

    return tl::make_unexpected(Error{std::error_code(),
                                     util::fmt("Warning: no handle for attribute {} for node {} \n",
                                               attr.getNormalizedFullName(),
                                               kind.asStringRef().str())
                                         .value()});
}

bool CommonAttributeMap::hasHandler(const KeyType& key) {
    auto it = _nodeHandlers.find(key);
    if (it != _nodeHandlers.cend()) {
        return true;
    }

    it = _nodeHandlers.find({std::get<0>(key), std::get<1>(key).getCladeKind()});
    return it != _nodeHandlers.cend();
}

}  // namespace oklt
