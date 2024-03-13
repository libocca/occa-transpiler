#pragma once

#include <llvm/Support/VirtualFileSystem.h>

namespace oklt {
struct TransformedFiles;
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeOverlayFs(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
    const TransformedFiles&);
}  // namespace oklt
