#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "core/attribute_manager/parsed_attribute_info_base.hpp"

namespace oklt {
using namespace clang;
using namespace llvm;

bool ParsedAttrInfoBase::MustDelayAttributeArguments(const ParsedAttr& A) const {
    // Only attributes that accept expression parameter packs can delay arguments.
    if (!A.acceptsExprPack()) {
        return false;
    }

    unsigned AttrNumArgs = A.getNumArgMembers();
    for (size_t I = 0; I < std::min(A.getNumArgs(), AttrNumArgs); ++I) {
        bool IsLastAttrArg = I == (AttrNumArgs - 1);
        if (IsLastAttrArg && hasVariadicArg()) {
            return false;
        }

        clang::Expr* E = A.getArgAsExpr(I);
        bool ArgMemberCanHoldExpr = A.isParamExpr(I);

        if (llvm::isa<clang::PackExpansionExpr>(E)) {
            return !(IsLastAttrArg && ArgMemberCanHoldExpr);
        }

        if (E->isValueDependent() && !ArgMemberCanHoldExpr) {
            return true;
        }
    }
    return false;
}

ParsedAttrInfoBase::AttrHandling ParsedAttrInfoBase::handleAttribute(Sema& S,
                                                                     Decl& D,
                                                                     const ParsedAttr& AL) const {
    StringRef Str;
    if (!S.checkStringLiteralArgumentAttr(AL, 0, Str)) {
        return AttributeNotApplied;
    }

    SmallVector<Expr*, 4> Args;
    Args.reserve(AL.getNumArgs() - 1);
    for (unsigned Idx = 1; Idx < AL.getNumArgs(); Idx++) {
        assert(!AL.isArgIdent(Idx));
        Args.push_back(AL.getArgAsExpr(Idx));
    }

    auto* ctxAttr = AnnotateAttr::Create(S.Context, Str, Args.data(), Args.size(), AL);
    return addAttr(S, D, ctxAttr);
}

ParsedAttrInfoBase::AttrHandling ParsedAttrInfoBase::handleAttribute(Sema& S,
                                                                     Stmt& St,
                                                                     const ParsedAttr& AL) const {
    if (!AL.checkAtLeastNumArgs(S, 1)) {
        return AttributeNotApplied;
    }

    std::vector<StringRef> Args;
    for (unsigned I = 0; I != AL.getNumArgs(); ++I) {
        StringRef ArgStr;
        if (!S.checkStringLiteralArgumentAttr(AL, I, ArgStr, nullptr)) {
            return AttributeNotApplied;
        }

        Args.push_back(ArgStr);
    }

    auto* ctxAttr = clang::SuppressAttr::Create(S.Context, Args.data(), Args.size(), AL);
    return addAttr(S, St, ctxAttr);
}

bool ParsedAttrInfoBase::diagAppertainsToDecl(Sema& S,
                                              const ParsedAttr& Attr,
                                              const Decl* D) const {
    bool MustDelayArgs = MustDelayAttributeArguments(Attr);
    return (D && !checkAttrFeatures(S, *D, Attr, MustDelayArgs));
}

bool ParsedAttrInfoBase::diagAppertainsToStmt(Sema& S,
                                              const ParsedAttr& Attr,
                                              const Stmt* St) const {
    bool MustDelayArgs = MustDelayAttributeArguments(Attr);
    if (St && !checkAttrFeatures(S, *St, Attr, MustDelayArgs)) {
        if (handleAttribute(S, const_cast<Stmt&>(*St), Attr) == ParsedAttrInfo::NotHandled) {
            S.Diag(Attr.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
                << Attr << Attr.isRegularKeywordAttribute() << St->getBeginLoc();
        }
    }
    return false;
}

bool ParsedAttrInfoBase::diagMutualExclusion(Sema& S, const ParsedAttr& A, const Decl* D) const {
    if (!D) {
        return true;
    }

    auto* stage = getStageFromASTContext(S.Context);
    if (!stage) {
        return true;
    }

    auto& attrStore = stage->tryEmplaceUserCtx<AttributeStore>(S.Context);
    return diagMutualExclusion(S, A, *D, attrStore.get(*D));
}

ParsedAttrInfoBase::AttrHandling
ParsedAttrInfoBase::handleDeclAttribute(Sema& S, Decl* D, const ParsedAttr& Attr) const {
    if (!D) {
        return NotHandled;
    }

    return handleAttribute(S, *D, Attr);
}

}  // namespace oklt
