#pragma once

#include "core/rewriter/rewriter_proxy.h"

namespace clang {
class SourceManager;
class LangOptions;
}  // namespace clang
namespace oklt {
enum class RewriterProxyType {
    Original,
    WithDeltaTree,
};

std::unique_ptr<RewriterProxy> makeRewriterProxy(clang::SourceManager&,
                                                 clang::LangOptions&,
                                                 RewriterProxyType);
}  // namespace oklt
