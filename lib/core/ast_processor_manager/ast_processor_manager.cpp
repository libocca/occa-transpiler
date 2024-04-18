#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace {
using namespace oklt;
using namespace clang;

template <typename MapType, typename KeysType, typename ActionFunc>
oklt::HandleResult runNodeHandler(MapType& map, KeysType& keys, ActionFunc action) {
    for (auto& key : keys) {
        auto it = map.find(key);
        if (it != map.end()) {
            return action(it->second);
        }
    }

    return {};
}

template <typename KeyType>
std::vector<KeyType> makeKeys(const clang::DynTypedNode& node, const clang::Attr* attr) {
    auto kind = node.getNodeKind();
    auto name = attr ? attr->getNormalizedFullName() : "";
    return {{name, kind}, {name, kind.getCladeKind()}};
}

}  // namespace

namespace oklt {
using namespace clang;

AstProcessorManager& AstProcessorManager::instance() {
    static AstProcessorManager manager;
    return manager;
}

bool AstProcessorManager::registerHandle(KeyType key, NodeHandle handle) {
    auto [_, ret] = _nodeHandlers.try_emplace(key, std::move(handle));
    return ret;
}

HandleResult AstProcessorManager::runPreActionNodeHandle(SessionStage& stage,
                                                         const clang::DynTypedNode& node,
                                                         const clang::Attr* attr) {
    auto keys = makeKeys<KeyType>(node, attr);
    return runNodeHandler(
        _nodeHandlers, keys, [&](auto h) { return h.preAction(stage, node, attr); });
}

HandleResult AstProcessorManager::runPostActionNodeHandle(SessionStage& stage,
                                                          const clang::DynTypedNode& node,
                                                          const clang::Attr* attr) {
    auto keys = makeKeys<KeyType>(node, attr);
    return runNodeHandler(
        _nodeHandlers, keys, [&](auto h) { return h.postAction(stage, node, attr); });
}

}  // namespace oklt
