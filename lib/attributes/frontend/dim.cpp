#include "params/dim.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "core/diag/diag_handler.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling DIM_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "dim"},
    {ParsedAttr::AS_CXX11, DIM_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_dim"}};

struct DimAttribute : public ParsedAttrInfoBase {
    DimAttribute() {
        Spellings = DIM_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        if (!isa<VarDecl, ParmVarDecl, TypeDecl, FieldDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute()
                << "type, struct/union/class field or variable declarations";
            return false;
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute()
            << "type, struct/union/class field or variable declarations";
        return false;
    }
};

class DimDiagHandler : public DiagHandler {
   public:
    DimDiagHandler()
        : DiagHandler(diag::err_typecheck_call_not_function){};

    bool HandleDiagnostic(SessionStage& session, DiagLevel level, const Diagnostic& info) override {
        if (info.getArgKind(0) != DiagnosticsEngine::ak_qualtype)
            return false;

        QualType qt = QualType::getFromOpaquePtr(reinterpret_cast<void*>(info.getRawArg(0)));

        static llvm::ManagedStatic<SmallVector<StringRef>> attrNames = {};
        if (attrNames->empty()) {
            for (auto v : DIM_ATTRIBUTE_SPELLINGS) {
                attrNames->push_back(v.NormalizedFullName);
            }
        };

        auto& ctx = session.getCompiler().getASTContext();
        auto& attrStore = session.tryEmplaceUserCtx<AttributeStore>(ctx);
        if (attrStore.has(qt, *attrNames))
            return true;

        return false;
    }
};

ParseResult parseDimAttrParams(const clang::Attr& attr, OKLParsedAttr& data, SessionStage& stage) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@dim] does not take kwargs"});
    }

    if (data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@dim] expects at least one argument"});
    }

    AttributedDim ret;
    ret.dim.reserve(data.args.size());
    for (auto arg : data.args) {
        ret.dim.emplace_back(arg.getRaw());
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<DimAttribute>(DIM_ATTR_NAME,
                                                                    parseDimAttrParams);
    // for suppression of func call error that potentially is dim calls
    static oklt::DiagHandlerRegistry::Add<DimDiagHandler> diag_dim("DimDiagHandler", "");
}
}  // namespace
