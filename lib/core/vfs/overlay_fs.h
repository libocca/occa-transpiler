#pragma once

#include <llvm/Support/VirtualFileSystem.h>

#include <map>

namespace oklt {
struct TransformedFiles;
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeOverlayFs(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
    const std::map<std::string, std::string>&);
}  // namespace oklt
