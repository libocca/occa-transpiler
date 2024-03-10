#pragma once

#include <llvm/Support/VirtualFileSystem.h>

namespace oklt {
struct TransformedHeaders;
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeOverlayFs(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
    const TransformedHeaders&);
}  // namespace oklt
