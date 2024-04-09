#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt {
using namespace clang;

AttributeManager& AttributeManager::instance() {
    static AttributeManager attrManager;
    return attrManager;
}

bool AttributeManager::registerCommonHandler(std::string name, AttrDeclHandler handler) {
    return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerCommonHandler(std::string name, AttrStmtHandler handler) {
    return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrDeclHandler handler) {
    return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrStmtHandler handler) {
    return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               DeclHandler handler) {
    return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               StmtHandler handler) {
    return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

HandleResult AttributeManager::handleNode(const Stmt& stmt, SessionStage& stage) {
    return _implicitHandlers(stmt, stage);
}

HandleResult AttributeManager::handleNode(const Decl& decl, SessionStage& stage) {
    return _implicitHandlers(decl, stage);
}

HandleResult AttributeManager::handleAttr(const Attr& attr,
                                          const Decl& decl,
                                          const std::any* params,
                                          SessionStage& stage) {
    std::string name = attr.getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
        return _commonAttrs.handleAttr(attr, decl, params, stage);
    }

    if (_backendAttrs.hasAttrHandler(stage, name)) {
        return _backendAttrs.handleAttr(attr, decl, params, stage);
    }

    // TODO: Uncomment after multi-handle added
    // return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    return {};
}

HandleResult AttributeManager::handleAttr(const Attr& attr,
                                          const Stmt& stmt,
                                          const std::any* params,
                                          SessionStage& stage) {
    std::string name = attr.getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
        return _commonAttrs.handleAttr(attr, stmt, params, stage);
    }

    if (_backendAttrs.hasAttrHandler(stage, name)) {
        return _backendAttrs.handleAttr(attr, stmt, params, stage);
    }

    // TODO: Uncomment after multi-handle added
    // return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
    return {};
}

ParseResult AttributeManager::parseAttr(const Attr& attr, SessionStage& stage) {
    std::string name = attr.getNormalizedFullName();
    auto it = _attrParsers.find(name);
    if (it != _attrParsers.end()) {
        auto params = ParseOKLAttr(attr, stage);
        return it->second(attr, params, stage);
    }

    return EmptyParams{};
}

ParseResult AttributeManager::parseAttr(const Attr& attr,
                                        OKLParsedAttr& params,
                                        SessionStage& stage) {
    auto it = _attrParsers.find(params.name);
    if (it != _attrParsers.end()) {
        return it->second(attr, params, stage);
    }
    return EmptyParams{};
}

tl::expected<std::set<const Attr*>, Error> AttributeManager::checkAttrs(const Decl& decl,
                                                                        SessionStage& stage) {
    if (!decl.hasAttrs()) {
        return {};
    }

    const auto& attrs = decl.getAttrs();
    std::set<const Attr*> collectedAttrs;

    // in case of multiple same attribute take the last one
    for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        const auto& attr = *it;

        if (!attr) {
            continue;
        }

        if (!isOklAttribute(*attr)) {
            continue;
        }

        auto name = attr->getNormalizedFullName();
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            SPDLOG_ERROR("{} attribute: {} for decl: {} doesn not have a registered handler",
                         decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                         name,
                         decl.getDeclKindName());

            return tl::make_unexpected(Error{.ec = std::error_code(), .message = "no handler"});
        }

        auto [_, isNew] = collectedAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            SPDLOG_ERROR("{} multi declaration of attribute: {} for decl: {}",
                         decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                         name,
                         decl.getDeclKindName());
        }
    }

    return collectedAttrs;
}

tl::expected<std::set<const Attr*>, Error> AttributeManager::checkAttrs(const Stmt& stmt,
                                                                        SessionStage& stage) {
    if (stmt.getStmtClass() != Stmt::AttributedStmtClass) {
        return {};
    }

    const auto& attrs = cast<AttributedStmt>(stmt).getAttrs();
    std::set<const Attr*> collectedAttrs;
    for (const auto attr : attrs) {
        if (!attr) {
            continue;
        }

        if (!isOklAttribute(*attr)) {
            continue;
        }

        auto name = attr->getNormalizedFullName();
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            SPDLOG_ERROR("{} attribute: {} for stmt: {} does not have a registered handler",
                         stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                         name,
                         stmt.getStmtClassName());
            return tl::make_unexpected(Error{.ec = std::error_code(), .message = "no handler"});
        }

        auto [_, isNew] = collectedAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            SPDLOG_ERROR("{} multi declaration of attribute: {} for stmt: {}",
                         stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                         name,
                         stmt.getStmtClassName());
        }
    }

    return collectedAttrs;
}
}  // namespace oklt
