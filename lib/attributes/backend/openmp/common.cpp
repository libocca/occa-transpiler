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

    for (auto& var : loopInfo.vars.exclusive) {
        auto loc = var.get().getLocation().getLocWithOffset(var.get().getName().size());
        trans.addReplacement(OKL_TRANSPILED_ATTR, loc, varSuffix);
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
