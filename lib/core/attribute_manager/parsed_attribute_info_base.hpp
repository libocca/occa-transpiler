#pragma once

#include "core/attribute_manager/parsed_attribute_info_base.h"

namespace oklt {

template <typename T>
ParsedAttrInfoBase::AttrHandling ParsedAttrInfoBase::addAttr(clang::Sema& S,
                                                             T& Node,
                                                             clang::Attr* Attr) const {
    if (!Attr) {
        return AttributeNotApplied;
    }

    auto* stage = getStageFromASTContext(S.Context);
    if (!stage) {
        return AttributeNotApplied;
    }

    auto& attrStore = stage->tryEmplaceUserCtx<AttributeStore>(S.Context);
    auto ret = attrStore.add<T>(Node, *Attr, AttrKind);
    if (!ret) {
        return AttributeNotApplied;
    }

    S.Context.addDestruction(Attr);

    if constexpr (std::is_base_of_v<clang::Decl, T>) {
        if (!IsType) {
            return AttributeApplied;
        }

        // Apply Attr to VarDecl
        if (auto pVarDecl = llvm::dyn_cast<clang::VarDecl>(&Node)) {
            clang::QualType origType = pVarDecl->getType();
            clang::QualType modifiedType =
                S.Context.getTypeOfType(origType, clang::TypeOfKind::Qualified);

            clang::QualType newType =
                S.Context.getAttributedType(clang::attr::AnnotateType, modifiedType, origType);
            pVarDecl->setType(newType);

            attrStore.add<clang::QualType>(newType, *Attr, AttrKind);
            return AttributeApplied;
        }

        // TypeDecl
        if (auto pTypeDecl = llvm::dyn_cast<clang::TypeDecl>(&Node)) {
            clang::QualType origType = S.Context.getTypeDeclType(pTypeDecl);
            clang::QualType newType =
                S.Context.getAttributedType(clang::attr::AnnotateType, origType, origType);
            pTypeDecl->setTypeForDecl(newType.getTypePtr());

            attrStore.add<clang::QualType>(newType, *Attr, AttrKind);
            return AttributeApplied;
        }

        return AttributeNotApplied;
    }

    return AttributeApplied;
}

template <typename T>
bool ParsedAttrInfoBase::checkAttrFeatures(clang::Sema& S,
                                           const T& Node,
                                           const ParsedAttr& A,
                                           bool SkipArgCountCheck) const {
    if (!diagAppertainsTo(S, A, Node))
        return true;

    auto* stage = getStageFromASTContext(S.Context);
    if (!stage) {
        return true;
    }

    auto& attrStore = stage->tryEmplaceUserCtx<AttributeStore>(S.Context);
    if (!diagMutualExclusion(S, A, Node, attrStore.get(Node))) {
        return true;
    }
    if (S.CheckAttrTarget(A)) {
        return true;
    }

    if (A.hasCustomParsing()) {
        return false;
    }

    if (!SkipArgCountCheck) {
        if (A.getMinArgs() == A.getMaxArgs()) {
            if (!A.checkExactlyNumArgs(S, A.getMinArgs()))
                return true;
        } else {
            if (A.getMinArgs() && !A.checkAtLeastNumArgs(S, A.getMinArgs())) {
                return true;
            }
            if (!A.hasVariadicArg() && A.getMaxArgs() && !A.checkAtMostNumArgs(S, A.getMaxArgs())) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace oklt
