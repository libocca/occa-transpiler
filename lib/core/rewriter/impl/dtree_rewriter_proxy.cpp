#include "core/rewriter/impl/dtree_rewriter_proxy.h"

namespace oklt {
DtreeRewriterProxy::DtreeRewriterProxy(clang::SourceManager& SM, const clang::LangOptions& LO)
    : _dtrees(SM, LO),
      RewriterProxy(SM, LO) {}

const DeltaTrees& DtreeRewriterProxy::getDeltaTrees() const {
    return _dtrees;
}

bool DtreeRewriterProxy::InsertText(clang::SourceLocation Loc,
                                    clang::StringRef Str,
                                    bool InsertAfter,
                                    bool indentNewLines) {
    _dtrees.Insert(Loc, Str.size(), InsertAfter);
    return RewriterProxy::InsertText(Loc, Str, InsertAfter, indentNewLines);
}

bool DtreeRewriterProxy::InsertTextAfterToken(clang::SourceLocation Loc, clang::StringRef Str) {
    _dtrees.Insert(Loc, Str.size(), true);
    return RewriterProxy::InsertTextAfterToken(Loc, Str);
}

bool DtreeRewriterProxy::RemoveText(clang::SourceLocation Start,
                                    unsigned Length,
                                    clang::Rewriter::RewriteOptions opts) {
    _dtrees.Remove(Start, Length);
    return RewriterProxy::RemoveText(Start, Length, opts);
}

bool DtreeRewriterProxy::ReplaceText(clang::SourceLocation Start,
                                     unsigned OrigLength,
                                     clang::StringRef NewStr) {
    _dtrees.Replace(Start, OrigLength, NewStr.size());
    return RewriterProxy::ReplaceText(Start, OrigLength, NewStr);
}
bool DtreeRewriterProxy::ReplaceText(clang::SourceRange range,
                                     clang::SourceRange replacementRange) {
    _dtrees.Replace(range.getBegin(), getRangeSize(range), getRangeSize(replacementRange));
    return RewriterProxy::ReplaceText(range, replacementRange);
}
}  // namespace oklt
