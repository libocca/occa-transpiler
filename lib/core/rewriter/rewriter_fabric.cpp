#include "core/rewriter/rewriter_fabric.h"

#include "core/rewriter/impl/dtree_rewriter_proxy.h"
#include "core/rewriter/impl/empty_rewriter_proxy.h"

namespace oklt {
std::unique_ptr<RewriterProxy> makeRewriterProxy(clang::SourceManager& SM,
                                                 clang::LangOptions& LO,
                                                 RewriterProxyType rtype) {
    switch (rtype) {
        case RewriterProxyType::Original: {
            return std::make_unique<EmptyRewriterProxy>(SM, LO);
        }
        case RewriterProxyType::WithDeltaTree: {
            return std::make_unique<DtreeRewriterProxy>(SM, LO);
        }
        default:
            return std::make_unique<EmptyRewriterProxy>(SM, LO);
    }
}
}  // namespace oklt
