#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attribute_store.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

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
    auto name = getOklAttrFullName(attr);
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
    auto name = getOklAttrFullName(attr);
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
    auto name = getOklAttrFullName(attr);
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

tl::expected<std::vector<const Attr*>, Error> AttributeManager::checkAttrs(const Decl& decl,
                                                                           SessionStage& stage) {
    auto& attrStore = stage.tryEmplaceUserCtx<AttributeStore>(stage.getCompiler().getASTContext());
    auto attrs = attrStore.get(decl);
    if (attrs.empty()) {
        return {};
    }

    std::vector<const Attr*> ret;
    ret.reserve(attrs.size());

    std::set<const Attr*> uniqueAttrs;
    for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        auto attr = *it;
        if (!attr) {
            continue;
        }

        auto name = getOklAttrFullName(*attr);
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            llvm::errs() << decl.getBeginLoc().printToString(
                                decl.getASTContext().getSourceManager())
                         << " attribute: " << name << " for decl: " << decl.getDeclKindName()
                         << " does not have registered handler \n";

            return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
        }

        auto [_, isNew] = uniqueAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            llvm::errs() << decl.getBeginLoc().printToString(
                                decl.getASTContext().getSourceManager())
                         << " multi declaration of attribute: " << name
                         << " for decl: " << decl.getDeclKindName() << '\n';
        }

        ret.push_back(attr);
    }

    return ret;
}

tl::expected<std::vector<const Attr*>, Error> AttributeManager::checkAttrs(const Stmt& stmt,
                                                                           SessionStage& stage) {
    auto& attrStore = stage.tryEmplaceUserCtx<AttributeStore>(stage.getCompiler().getASTContext());
    auto attrs = attrStore.get(stmt);
    if (attrs.empty()) {
        return {};
    }

    std::vector<const Attr*> ret;
    ret.reserve(attrs.size());

    std::set<const Attr*> uniqueAttrs;
    for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
        auto attr = *it;
        if (!attr) {
            continue;
        }

        auto name = getOklAttrFullName(*attr);
        if (!_commonAttrs.hasAttrHandler(name) && !_backendAttrs.hasAttrHandler(stage, name)) {
            // TODO report diag error
            llvm::errs() << stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager())
                         << " attribute: " << name << " for stmt: " << stmt.getStmtClassName()
                         << " does not have registered handler \n";

            return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
        }

        auto [_, isNew] = uniqueAttrs.insert(attr);
        if (!isNew) {
            // TODO convince OCCA community to specify such case as forbidden
            llvm::errs() << stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager())
                         << " multi declaration of attribute: " << name
                         << " for stmt: " << stmt.getStmtClassName() << '\n';
        }

        ret.push_back(attr);
    }

    return ret;
}
}  // namespace oklt
