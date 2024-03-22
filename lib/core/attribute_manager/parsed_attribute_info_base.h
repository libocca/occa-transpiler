#pragma once

#include "attribute_manager.h"
#include "attribute_store.h"
#include "core/transpiler_session/session_stage.h"

#include "oklt/util/string_utils.h"

#include <clang/AST/Attr.h>
#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaDiagnostic.h>

namespace oklt {

class ParsedAttrInfoBase : public clang::ParsedAttrInfo {
   public:
    typedef clang::AttributeCommonInfo::Kind Kind;
    typedef clang::ParsedAttr ParsedAttr;
    typedef ParsedAttrInfo::AttrHandling AttrHandling;

    AttributeKind AttrKind = AttributeKind::COMMON;

    /// Check if this attribute appertains to D, and issue a diagnostic if not.
    virtual bool diagAppertainsTo(clang::Sema& S,
                                  const ParsedAttr& Attr,
                                  const clang::Decl& D) const {
        return false;
    };

    /// Check if this attribute appertains to St, and issue a diagnostic if not.
    virtual bool diagAppertainsTo(clang::Sema& S,
                                  const ParsedAttr& Attr,
                                  const clang::Stmt& St) const {
        return false;
    };

    /// Check if the given attribute is mutually exclusive with other attributes
    /// already applied to the given declaration.
    virtual bool diagMutualExclusion(clang::Sema& S,
                                     const ParsedAttr& A,
                                     const clang::Decl& D,
                                     const clang::AttrVec& Attrs) const {
        return true;
    }

    /// Check if the given attribute is mutually exclusive with other attributes
    /// already applied to the given statement.
    virtual bool diagMutualExclusion(clang::Sema& S,
                                     const ParsedAttr& A,
                                     const clang::Stmt& St,
                                     const clang::AttrVec& Attrs) const {
        return true;
    }

    /// If this ParsedAttrInfo knows how to handle this ParsedAttr applied to this
    /// Decl then do so and return either AttributeApplied if it was applied or
    /// AttributeNotApplied if it wasn't. Otherwise return NotHandled.
    virtual AttrHandling handleAttribute(clang::Sema& S,
                                         clang::Decl& D,
                                         const ParsedAttr& AL) const;

    /// If this ParsedAttrInfo knows how to handle this ParsedAttr applied to this
    /// Stmt then do so and return either AttributeApplied if it was applied or
    /// AttributeNotApplied if it wasn't. Otherwise return NotHandled.
    virtual AttrHandling handleAttribute(clang::Sema& S,
                                         clang::Stmt& St,
                                         const ParsedAttr& AL) const;

    bool diagAppertainsToDecl(clang::Sema& S,
                              const ParsedAttr& Attr,
                              const clang::Decl* D) const final;

    bool diagAppertainsToStmt(clang::Sema& S,
                              const ParsedAttr& Attr,
                              const clang::Stmt* St) const final;

    bool diagMutualExclusion(clang::Sema& S, const ParsedAttr& A, const clang::Decl* D) const final;

    AttrHandling handleDeclAttribute(clang::Sema& S,
                                     clang::Decl* D,
                                     const ParsedAttr& Attr) const final;

   protected:
    template <typename T>
    AttrHandling addAttr(clang::Sema& S, T& Node, clang::Attr* Attr) const;

   private:
    [[nodiscard]] inline bool hasVariadicArg() const { return OptArgs == 15; }
    [[nodiscard]] inline unsigned getMaxArgs() const { return NumArgs + OptArgs; }

    [[nodiscard]] bool MustDelayAttributeArguments(const ParsedAttr& A) const;

    template <typename T>
    bool checkAttrFeatures(clang::Sema& S,
                           const T& Node,
                           const ParsedAttr& A,
                           bool SkipArgCountCheck) const;
};

}  // namespace oklt
