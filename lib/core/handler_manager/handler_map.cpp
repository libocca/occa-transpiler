#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "core/handler_manager/backend_handler.h"
#include "core/handler_manager/handler_map.h"
#include "core/handler_manager/implicid_handler.h"
#include "core/handler_manager/parse_handler.h"
#include "core/handler_manager/sema_handler.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

// Implicit
HandleResult HandlerMap::operator()(SessionStage& stage, const clang::DynTypedNode& node) {
    auto backend = stage.getBackend();
    auto kind = node.getNodeKind();

    for (auto& key : std::array{
             HandlerKey<HandleType::IMPLICIT>{backend, kind},
             HandlerKey<HandleType::IMPLICIT>{backend, kind.getCladeKind()},
         }) {
        auto it = _nodeHandlers.find(key);
        if (it != _nodeHandlers.end()) {
            return it->second->handle(stage, node);
        }
    }

    // INFO: implicit handler means that only some specific stmt/decl has specific handler
    //       missing of handler is ok
    return {};
}

// Common & Backend
HandleResult HandlerMap::operator()(oklt::SessionStage& stage,
                                    const clang::DynTypedNode& node,
                                    const clang::Attr& attr,
                                    const std::any* params) {
    auto backend = stage.getBackend();
    auto name = attr.getNormalizedFullName();
    auto kind = node.getNodeKind();

    // Common
    for (auto& key : std::array{
             HandlerKey<HandleType::COMMON>{name, kind},
             HandlerKey<HandleType::COMMON>{name, kind.getCladeKind()},
         }) {
        auto it = _nodeHandlers.find(key);
        if (it != _nodeHandlers.end()) {
            return it->second->handle(stage, node, attr, params);
        }
    }

    // Backend
    for (auto& key : std::array{
             HandlerKey<HandleType::BACKEND>{backend, name, kind},
             HandlerKey<HandleType::BACKEND>{backend, name, kind.getCladeKind()},
         }) {
        auto it = _nodeHandlers.find(key);
        if (it != _nodeHandlers.end()) {
            return it->second->handle(stage, node, attr, params);
        }
    }

    return tl::make_unexpected(
        Error{std::error_code(),
              util::fmt("Warning: no handle for backend {} for attribute {} for node {} \n",
                        backendToString(backend),
                        attr.getNormalizedFullName(),
                        kind.asStringRef().str())
                  .value()});
}

// Parser
HandleResult HandlerMap::operator()(oklt::SessionStage& stage,
                                    const clang::Attr& attr,
                                    OKLParsedAttr* params) {
    auto name = params ? params->name : attr.getNormalizedFullName();
    auto key = HandlerKey<HandleType::PARSER>{name};

    auto it = _nodeHandlers.find(key);
    if (it != _nodeHandlers.end()) {
        if (!params) {
            auto p = ParseOKLAttr(stage, attr);
            return it->second->handle(stage, attr, p);
        }
        return it->second->handle(stage, attr, *params);
    }

    return EmptyParams{};
}

// Sema (Pre)
HandleResult HandlerMap::pre(SessionStage& stage,
                             const clang::DynTypedNode& node,
                             const clang::Attr* attr) {
    auto name = attr ? attr->getNormalizedFullName() : "";
    auto kind = node.getNodeKind();

    // Sema
    for (auto& key : std::array{
             HandlerKey<HandleType::SEMA>{name, kind},
             HandlerKey<HandleType::SEMA>{name, kind.getCladeKind()},
         }) {
        auto it = _nodeHandlers.find(key);
        if (it != _nodeHandlers.end()) {
            return it->second->pre(stage, node, attr);
        }
    }

    return {};
}

// Sema (Post)
HandleResult HandlerMap::post(SessionStage& stage,
                              const clang::DynTypedNode& node,
                              const clang::Attr* attr) {
    auto name = attr ? attr->getNormalizedFullName() : "";
    auto kind = node.getNodeKind();

    // Sema
    for (auto& key : std::array{
             HandlerKey<HandleType::SEMA>{name, kind},
             HandlerKey<HandleType::SEMA>{name, kind.getCladeKind()},
         }) {
        auto it = _nodeHandlers.find(key);
        if (it != _nodeHandlers.end()) {
            return it->second->post(stage, node, attr);
        }
    }

    return {};
}

// Common
bool HandlerMap::hasHandler(const std::string& name, clang::ASTNodeKind kind) const {
    for (auto& key : std::array{
             HandlerKey<HandleType::COMMON>{name, kind},
             HandlerKey<HandleType::COMMON>{name, kind.getCladeKind()},
         }) {
        if (_nodeHandlers.find(key) != _nodeHandlers.end()) {
            return true;
        }
    }

    return false;
}

// Backend
bool HandlerMap::hasHandler(const TargetBackend backend,
                            const std::string& name,
                            clang::ASTNodeKind kind) const {
    for (auto& key : std::array{
             HandlerKey<HandleType::BACKEND>{backend, name, kind},
             HandlerKey<HandleType::BACKEND>{backend, name, kind.getCladeKind()},
         }) {
        if (_nodeHandlers.find(key) != _nodeHandlers.end()) {
            return true;
        }
    }

    return false;
}

// Implicit
bool HandlerMap::hasHandler(const TargetBackend backend, clang::ASTNodeKind kind) const {
    for (auto& key : std::array{
             HandlerKey<HandleType::IMPLICIT>{backend, kind},
             HandlerKey<HandleType::IMPLICIT>{backend, kind.getCladeKind()},
         }) {
        if (_nodeHandlers.find(key) != _nodeHandlers.end()) {
            return true;
        }
    }

    return false;
}

// Parser
bool HandlerMap::hasHandler(const std::string& name) const {
    auto key = HandlerKey<HandleType::PARSER>{name};
    return (_nodeHandlers.find(key) != _nodeHandlers.end());
}

// Sema
bool HandlerMap::hasSemeHandler(const std::string& name, clang::ASTNodeKind kind) const {
    for (auto& key : std::array{
             HandlerKey<HandleType::SEMA>{name, kind},
             HandlerKey<HandleType::SEMA>{name, kind.getCladeKind()},
         }) {
        if (_nodeHandlers.find(key) != _nodeHandlers.end()) {
            return true;
        }
    }

    return false;
}

}  // namespace oklt
