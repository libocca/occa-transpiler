#include "core/rewriter/rewriter_proxy.h"

namespace oklt {
RewriterProxy::RewriterProxy(clang::SourceManager& SM, const clang::LangOptions& LO)
    : _rewriter(SM, LO) {}

void RewriterProxy::setSourceMgr(clang::SourceManager& SM, const clang::LangOptions& LO) {
    return _rewriter.setSourceMgr(SM, LO);
}

clang::SourceManager& RewriterProxy::getSourceMgr() const {
    return _rewriter.getSourceMgr();
}
const clang::LangOptions& RewriterProxy::getLangOpts() const {
    return _rewriter.getLangOpts();
}

int RewriterProxy::getRangeSize(clang::SourceRange Range,
                                clang::Rewriter::RewriteOptions opts) const {
    return _rewriter.getRangeSize(Range, opts);
}
int RewriterProxy::getRangeSize(const clang::CharSourceRange& Range,
                                clang::Rewriter::RewriteOptions opts) const {
    return _rewriter.getRangeSize(Range, opts);
}

std::string RewriterProxy::getRewrittenText(clang::CharSourceRange Range) const {
    return _rewriter.getRewrittenText(Range);
}

bool RewriterProxy::InsertText(clang::SourceLocation Loc,
                               clang::StringRef Str,
                               bool InsertAfter,
                               bool indentNewLines) {
    return _rewriter.InsertText(Loc, Str, InsertAfter, indentNewLines);
}

bool RewriterProxy::InsertTextAfterToken(clang::SourceLocation Loc, clang::StringRef Str) {
    return _rewriter.InsertTextAfterToken(Loc, Str);
}

bool RewriterProxy::RemoveText(clang::SourceLocation Start,
                               unsigned Length,
                               clang::Rewriter::RewriteOptions opts) {
    return _rewriter.RemoveText(Start, Length, opts);
}

bool RewriterProxy::ReplaceText(clang::SourceLocation Start,
                                unsigned OrigLength,
                                clang::StringRef NewStr) {
    return _rewriter.ReplaceText(Start, OrigLength, NewStr);
}
bool RewriterProxy::ReplaceText(clang::SourceRange range, clang::SourceRange replacementRange) {
    return _rewriter.ReplaceText(range, replacementRange);
}
bool RewriterProxy::IncreaseIndentation(clang::CharSourceRange range,
                                        clang::SourceLocation parentIndent) {
    return _rewriter.IncreaseIndentation(range, parentIndent);
}
clang::RewriteBuffer& RewriterProxy::getEditBuffer(clang::FileID FID) {
    return _rewriter.getEditBuffer(FID);
}

const clang::RewriteBuffer* RewriterProxy::getRewriteBufferFor(clang::FileID FID) const {
    return _rewriter.getRewriteBufferFor(FID);
}

clang::Rewriter::buffer_iterator RewriterProxy::buffer_begin() {
    return _rewriter.buffer_begin();
}
clang::Rewriter::buffer_iterator RewriterProxy::buffer_end() {
    return _rewriter.buffer_end();
}
clang::Rewriter::const_buffer_iterator RewriterProxy::buffer_begin() const {
    return _rewriter.buffer_begin();
}
clang::Rewriter::const_buffer_iterator RewriterProxy::buffer_end() const {
    return _rewriter.buffer_end();
}

bool RewriterProxy::overwriteChangedFiles() {
    return _rewriter.overwriteChangedFiles();
}

}  // namespace oklt
