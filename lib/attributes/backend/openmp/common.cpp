#include "attributes/backend/openmp/common.h"
#include "attributes/frontend/params/loop.h"
#include "core/transpilation_encoded_names.h"

namespace oklt::openmp {
using namespace oklt;
using namespace clang;

HandleResult postHandleExclusive(OklLoopInfo& loopInfo, TranspilationBuilder& trans) {
    if (loopInfo.vars.exclusive.empty()) {
        return {};
    }

    size_t sz = 0;
    for (auto child : loopInfo.children) {
        auto v = child.getSize();
        if (!v.has_value()) {
            sz = 1024;
            break;
        }
        sz = std::max(v.value(), sz);
    }
    std::string varSuffix = "[" + std::to_string(sz) + "]";

    for (auto& v : loopInfo.vars.exclusive) {
        auto& decl = v.get();
        auto nameLoc = decl.getLocation().getLocWithOffset(decl.getName().size());
        trans.addReplacement(OKL_TRANSPILED_ATTR, nameLoc, varSuffix);
        if (decl.hasInit()) {
            auto expr = decl.getInit();
            trans.addReplacement(OKL_TRANSPILED_ATTR, expr->getBeginLoc(), "{");
            trans.addReplacement(OKL_TRANSPILED_ATTR, decl.getEndLoc().getLocWithOffset(1), "}");
        }
    }

    return {};
}

HandleResult postHandleShared(OklLoopInfo& loopInfo, TranspilationBuilder& trans) {
    if (loopInfo.vars.shared.empty())
        return {};

    auto child = loopInfo.getFirstAttributedChild();
    if (!loopInfo.metadata.isOuter() || !child || child->metadata.type != LoopMetaType::Inner) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    return {};
}

}  // namespace oklt::openmp