#pragma once

#include <clang/Rewrite/Core/Rewriter.h>

namespace oklt {

// Simply redirects all calls to rewriter
class RewriterProxy {
   protected:
    clang::Rewriter _rewriter;

   public:
    explicit RewriterProxy() = default;
    explicit RewriterProxy(clang::SourceManager& SM, const clang::LangOptions& LO);

    virtual void setSourceMgr(clang::SourceManager& SM, const clang::LangOptions& LO);

    virtual clang::SourceManager& getSourceMgr() const;
    virtual const clang::LangOptions& getLangOpts() const;

    static bool isRewritable(clang::SourceLocation Loc) { return Loc.isFileID(); }

    virtual int getRangeSize(
        clang::SourceRange Range,
        clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions()) const;
    virtual int getRangeSize(
        const clang::CharSourceRange& Range,
        clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions()) const;

    virtual std::string getRewrittenText(clang::CharSourceRange Range) const;

    std::string getRewrittenText(clang::SourceRange Range) const {
        return getRewrittenText(clang::CharSourceRange::getTokenRange(Range));
    }

    virtual bool InsertText(clang::SourceLocation Loc,
                            clang::StringRef Str,
                            bool InsertAfter = true,
                            bool indentNewLines = false);

    bool InsertTextAfter(clang::SourceLocation Loc, clang::StringRef Str) {
        return InsertText(Loc, Str);
    }

    virtual bool InsertTextAfterToken(clang::SourceLocation Loc, clang::StringRef Str);

    bool InsertTextBefore(clang::SourceLocation Loc, clang::StringRef Str) {
        return InsertText(Loc, Str, false);
    }

    /// RemoveText - Remove the specified text region.
    virtual bool RemoveText(
        clang::SourceLocation Start,
        unsigned Length,
        clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions());

    /// Remove the specified text region.
    bool RemoveText(clang::CharSourceRange range,
                    clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions()) {
        return RemoveText(range.getBegin(), getRangeSize(range, opts), opts);
    }

    bool RemoveText(clang::SourceRange range,
                    clang::Rewriter::RewriteOptions opts = clang::Rewriter::RewriteOptions()) {
        return RemoveText(range.getBegin(), getRangeSize(range, opts), opts);
    }

    virtual bool ReplaceText(clang::SourceLocation Start,
                             unsigned OrigLength,
                             clang::StringRef NewStr);

    bool ReplaceText(clang::CharSourceRange range, clang::StringRef NewStr) {
        return ReplaceText(range.getBegin(), getRangeSize(range), NewStr);
    }

    bool ReplaceText(clang::SourceRange range, clang::StringRef NewStr) {
        return ReplaceText(range.getBegin(), getRangeSize(range), NewStr);
    }

    virtual bool ReplaceText(clang::SourceRange range, clang::SourceRange replacementRange);

    virtual bool IncreaseIndentation(clang::CharSourceRange range,
                                     clang::SourceLocation parentIndent);
    bool IncreaseIndentation(clang::SourceRange range, clang::SourceLocation parentIndent) {
        return IncreaseIndentation(clang::CharSourceRange::getTokenRange(range), parentIndent);
    }

    virtual clang::RewriteBuffer& getEditBuffer(clang::FileID FID);

    virtual const clang::RewriteBuffer* getRewriteBufferFor(clang::FileID FID) const;

    virtual clang::Rewriter::buffer_iterator buffer_begin();
    virtual clang::Rewriter::buffer_iterator buffer_end();
    virtual clang::Rewriter::const_buffer_iterator buffer_begin() const;
    virtual clang::Rewriter::const_buffer_iterator buffer_end() const;

    virtual bool overwriteChangedFiles();
};

using Rewriter = RewriterProxy;

}  // namespace oklt
