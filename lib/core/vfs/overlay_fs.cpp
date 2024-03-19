#include "core/vfs/overlay_fs.h"
#include "core/transpiler_session/header_info.h"

namespace oklt {
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeOverlayFs(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFs,
    const TransformedFiles& files) {
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlayFs(
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inMemoryFs(
        new llvm::vfs::InMemoryFileSystem);

    //  ovelay is FS stack - ORDER MATTER
    overlayFs->pushOverlay(baseFs);
    overlayFs->pushOverlay(inMemoryFs);

    for (const auto& f : files.fileMap) {
        printf("add file: %s with conent:\n%s\n", f.first.c_str(), f.second.c_str());
        inMemoryFs->addFile(f.first, 0, llvm::MemoryBuffer::getMemBuffer(f.second));
    }

    return overlayFs;
}
}  // namespace oklt
