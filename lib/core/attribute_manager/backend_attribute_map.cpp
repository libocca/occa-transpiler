#include <oklt/util/string_utils.h>

#include "core/attribute_manager/backend_attribute_map.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool BackendAttributeMap::registerHandler(KeyType key, AttrHandler handler) {
    auto ret = _nodeHandlers.emplace(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

HandleResult BackendAttributeMap::handleAttr(SessionStage& stage,
                                             const clang::DynTypedNode& node,
                                             const clang::Attr& attr,
                                             const std::any* params) {
    auto backend = stage.getBackend();
    auto kind = node.getNodeKind();
    auto name = attr.getNormalizedFullName();

    auto it = _nodeHandlers.find({backend, name, kind});
    if (it != _nodeHandlers.end()) {
        return it->second.handle(stage, node, attr, params);
    }

    it = _nodeHandlers.find({backend, name, kind.getCladeKind()});
    if (it != _nodeHandlers.end()) {
        return it->second.handle(stage, node, attr, params);
    }

    return tl::make_unexpected(
        Error{std::error_code(),
              util::fmt("Warning: no handle for backend {} for attribute {} for node {} \n",
                        backendToString(backend),
                        attr.getNormalizedFullName(),
                        kind.asStringRef().str())
                  .value()});
}

bool BackendAttributeMap::hasHandler(const KeyType& key) {
    auto it = _nodeHandlers.find(key);
    if (it != _nodeHandlers.cend()) {
        return true;
    }

    it = _nodeHandlers.find({std::get<0>(key), std::get<1>(key), std::get<2>(key).getCladeKind()});
    return it != _nodeHandlers.cend();
}

}  // namespace oklt
