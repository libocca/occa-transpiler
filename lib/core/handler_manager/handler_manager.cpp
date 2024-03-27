#include "core/handler_manager/handler_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {

HandlerManager& HandlerManager::instance() {
    static HandlerManager hm;
    return hm;
}

bool HandlerManager::registerHandler(KeyType k, HandlerType h) {
    auto [_, ok] = _handlerMap.insert({k, std::move(h)});
    return ok;
}

bool HandlerManager::hasHandler(KeyType k) const {
    return _handlerMap.count(k);
}

HandlerManager::HandlerType* HandlerManager::tryGetHandler(KeyType k) {
    auto it = _handlerMap.find(k);
    if (it == _handlerMap.end()) {
        return nullptr;
    }

    return &it->second;
}

HandleResult HandlerManager::runHandler(KeyType k,
                                        const clang::Attr* a,
                                        const clang::DynTypedNode n,
                                        SessionStage& s) {
    auto* h = tryGetHandler(k);
    if (!h) {
        return tl::make_unexpected(Error{std::error_code(), "no handler under the key"});
    }

    return (*h)(a, n, s);
}

}  // namespace oklt
