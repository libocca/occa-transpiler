#pragma once

#include "core/rewriter/delta/delta_trees.h"
#include "core/rewriter/rewriter_proxy.h"

namespace oklt {
class DtreeRewriterProxy : public RewriterProxy {
    using RewriterProxy::RewriterProxy;

   private:
    DeltaTrees _dtrees;

   public:
    explicit DtreeRewriterProxy(clang::SourceManager& SM, const clang::LangOptions& LO);

    const DeltaTrees& getDeltaTrees() const;

    bool InsertText(clang::SourceLocation Loc,
                    clang::StringRef Str,
                    bool InsertAfter = true,
                    bool indentNewLines = false) override;
    bool InsertTextAfterToken(clang::SourceLocation Loc, clang::StringRef Str) override;
    bool RemoveText(
        clang::SourceLocation Start,
        unsigned Length,
        clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions()) override;

    bool ReplaceText(clang::SourceLocation Start,
                     unsigned OrigLength,
                     clang::StringRef NewStr) override;
    bool ReplaceText(clang::SourceRange range, clang::SourceRange replacementRange) override;
};
}  // namespace oklt
