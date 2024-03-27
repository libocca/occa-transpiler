#pragma once
#include <oklt/core/target_backends.h>

#include "core/handler_manager/categoty.h"
#include "core/handler_manager/handler_manager.h"
#include "util/type_traits.h"

#include "core/handler_manager/details/hashMaker.h"

namespace oklt {

template <typename HandlerType>
auto registerAttributedBackendHandler(TargetBackend backend,
                                      std::string_view attrName,
                                      HandlerType& handler) {
    using NodeType = typename std::remove_const_t<
        typename std::remove_reference_t<typename func_param_type<HandlerType, 2>::type>>;

    auto& hm = HandlerManager::instance();
    auto key = hm::detail::makeHash(HandlerCategory::BACKEND, backend, attrName);

    return hm.registerHandler(
        key, [&handler](const clang::Attr* a, const clang::DynTypedNode n, SessionStage& s) {
            return handler(*a, *(n.get<NodeType>()), s);
        });
}

}  // namespace oklt
