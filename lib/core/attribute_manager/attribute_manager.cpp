#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

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

    return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
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

    return tl::make_unexpected(Error{std::error_code(), "no handler for attr: " + name});
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

tl::expected<const Attr*, Error> AttributeManager::checkAttrs(const AttrVec& attrs,
                                                              const Decl& decl,
                                                              SessionStage& stage) {
    std::vector<const Attr*> collectedAttrs;
    collectedAttrs.reserve(attrs.size());
    for (const auto attr : attrs) {
        if (!attr)
            continue;

        auto name = attr->getNormalizedFullName();
        if (_commonAttrs.hasAttrHandler(name)) {
            collectedAttrs.push_back(attr);
            continue;
        }
        if (_backendAttrs.hasAttrHandler(stage, name)) {
            collectedAttrs.push_back(attr);
            continue;
        }
    }

    // INFO: there are no OKL attributes at all
    //       might need better solution for this
    if (collectedAttrs.empty()) {
        return {nullptr};
    }

    if (collectedAttrs.size() > 1) {
        return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "multiple attr"});
    }

    const Attr* attr = collectedAttrs.front();

    return attr;
}

tl::expected<const Attr*, Error> AttributeManager::checkAttrs(const ArrayRef<const Attr*>& attrs,
                                                              const Stmt& decl,
                                                              SessionStage& stage) {
    std::vector<const Attr*> collectedAttrs;
    collectedAttrs.reserve(attrs.size());
    for (const auto attr : attrs) {
        if (!attr)
            continue;

        auto name = attr->getNormalizedFullName();
        if (_commonAttrs.hasAttrHandler(name)) {
            collectedAttrs.push_back(attr);
            continue;
        }
        if (_backendAttrs.hasAttrHandler(stage, name)) {
            collectedAttrs.push_back(attr);
            continue;
        }
    }

    // INFO: there are no OKL attributes at all
    //       might need better solution for this
    if (collectedAttrs.empty()) {
        return nullptr;
    }

    if (collectedAttrs.size() > 1) {
        return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "multiple attr"});
    }

    const Attr* attr = collectedAttrs.front();

    return attr;
}
}  // namespace oklt
