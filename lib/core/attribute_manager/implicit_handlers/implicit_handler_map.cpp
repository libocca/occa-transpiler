#include "core/attribute_manager/implicit_handlers/implicit_handler_map.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool ImplicitHandlerMap::registerHandler(KeyType key, NodeHandler handler) {
    auto ret = _nodeHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
    return ret.second;
}

HandleResult ImplicitHandlerMap::operator()(SessionStage& stage, const DynTypedNode& node) {
    auto backend = stage.getBackend();
    auto kind = node.getNodeKind();
    auto it = _nodeHandlers.find({backend, kind});
    if (it != _nodeHandlers.end()) {
        return it->second(stage, node);
    }

    it = _nodeHandlers.find({backend, kind.getCladeKind()});
    if (it != _nodeHandlers.end()) {
        it->second(stage, node);
    }

    // INFO: implicit handler means that only some specific stmt/decl has specific handler
    //       missing of handler is ok
    return {};
}

bool ImplicitHandlerMap::hasHandler(const KeyType& key) const {
    auto it = _nodeHandlers.find(key);
    if (it != _nodeHandlers.cend()) {
        return true;
    }

    it = _nodeHandlers.find({std::get<0>(key), std::get<1>(key).getCladeKind()});
    return it != _nodeHandlers.cend();
}

}  // namespace oklt
