#include <oklt/core/error.h>

#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"
#include "core/handler_manager/handler_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <spdlog/spdlog.h>

namespace oklt {
using namespace clang;

HandlerManager& HandlerManager::instance() {
    static HandlerManager attrManager;
    return attrManager;
}

tl::expected<std::set<const Attr*>, Error> HandlerManager::checkAttrs(SessionStage& stage,
                                                                      const DynTypedNode& node) {
    auto backend = stage.getBackend();
    auto kind = node.getNodeKind();
    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(kind)) {
        auto& decl = node.getUnchecked<Decl>();
        if (!decl.hasAttrs()) {
            return {};
        }

        const auto& attrs = decl.getAttrs();

        // in case of multiple same attribute take the last one
        std::set<const Attr*> collectedAttrs;
        for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
            const auto& attr = *it;

            if (!attr || !isOklAttribute(*attr)) {
                continue;
            }

            auto name = attr->getNormalizedFullName();
            if (!_handlers.hasHandler(name, kind) && !_handlers.hasHandler(backend, name, kind)) {
                // TODO report diag error
                SPDLOG_ERROR(
                    "{} attribute: {} for decl: {} does not have a registered handler",
                    decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                    name,
                    decl.getDeclKindName());

                return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
            }

            auto [_, isNew] = collectedAttrs.insert(attr);
            if (!isNew) {
                // TODO convince OCCA community to specify such case as forbidden
                SPDLOG_ERROR(
                    "{} multi declaration of attribute: {} for decl: {}",
                    decl.getBeginLoc().printToString(decl.getASTContext().getSourceManager()),
                    name,
                    decl.getDeclKindName());
            }
        }

        return collectedAttrs;
    }

    if (ASTNodeKind::getFromNodeKind<AttributedStmt>().isBaseOf(kind)) {
        auto& stmt = node.getUnchecked<AttributedStmt>();
        auto subKind = ASTNodeKind::getFromNode(*stmt.getSubStmt());

        const auto& attrs = cast<AttributedStmt>(stmt).getAttrs();
        std::set<const Attr*> collectedAttrs;
        for (const auto attr : attrs) {
            if (!attr || !isOklAttribute(*attr)) {
                continue;
            }

            auto name = attr->getNormalizedFullName();
            if (!_handlers.hasHandler(name, subKind) &&
                !_handlers.hasHandler(backend, name, subKind)) {
                // TODO report diag error
                SPDLOG_ERROR(
                    "{} attribute: {} for stmt: {} does not have a registered handler",
                    stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                    name,
                    stmt.getStmtClassName());
                return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "no handler"});
            }

            auto [_, isNew] = collectedAttrs.insert(attr);
            if (!isNew) {
                // TODO convince OCCA community to specify such case as forbidden
                SPDLOG_ERROR(
                    "{} multi declaration of attribute: {} for stmt: {}",
                    stmt.getBeginLoc().printToString(stage.getCompiler().getSourceManager()),
                    name,
                    stmt.getStmtClassName());
            }
        }

        return collectedAttrs;
    }

    return {};
}

}  // namespace oklt
