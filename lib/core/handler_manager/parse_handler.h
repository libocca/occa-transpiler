#pragma once

#include "core/handler_manager/handler_manager.h"
#include "util/type_traits.h"

#include <clang/AST/ASTTypeTraits.h>
#include <clang/AST/Attr.h>
#include <tl/expected.hpp>

#include <any>
#include <functional>

namespace oklt {

class SessionStage;

class ParseHandler : public NodeHandler {
   public:
    using HandlerType =
        std::function<HandleResult(SessionStage&, const clang::Attr&, OKLParsedAttr&)>;

    explicit ParseHandler(HandlerType func)
        : handler(std::move(func)){};

    HandleResult handle(SessionStage& stage,
                        const clang::Attr& attr,
                        OKLParsedAttr& params) override {
        return handler(stage, attr, params);
    }

   private:
    const HandlerType handler;
};

template <enum HandleType T>
struct HandlerKey<T, std::enable_if_t<T == HandleType::PARSER>> : public HandleKeyBase {
    typedef ParseHandler HandlerType;

    template <typename... Ts>
    explicit HandlerKey(Ts&&... params)
        : HandleKeyBase(T) {
        using params_t = std::tuple<Ts&&...>;
        static_assert(std::is_same_v<std::decay_t<std::tuple_element_t<0, params_t>>, std::string>);

        auto p = std::forward_as_tuple(params...);
        attr = std::get<0>(p);
        if constexpr (std::tuple_size_v<params_t> == 2)
            if constexpr (std::is_same_v<std::decay_t<std::tuple_element_t<1, params_t>>,
                                         clang::ASTNodeKind>) {
                kind = std::get<1>(p);
            }
    }
};

template <typename AttrFrontendType, typename F>
inline bool HandlerManager::registerAttrFrontend(std::string attr, F& func) {
    static clang::ParsedAttrInfoRegistry::Add<AttrFrontendType> register_okl_atomic(attr, "");
    return _map().insert(HandlerKey<HandleType::PARSER>(attr), func);
};

}  // namespace oklt
