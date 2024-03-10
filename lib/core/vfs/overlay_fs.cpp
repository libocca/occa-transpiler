#include "core/vfs/overlay_fs.h"
#include "core/transpiler_session/header_info.h"

namespace oklt {
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> makeOverlayFs(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFs,
    const TransformedHeaders& headers) {
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlayFs(
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inMemoryFs(
        new llvm::vfs::InMemoryFileSystem);

    //  ovelay is FS stack - ORDER MATTER
    overlayFs->pushOverlay(baseFs);
    overlayFs->pushOverlay(inMemoryFs);

    for (const auto& f : headers.fileMap) {
        llvm::outs() << "overlayFs add file: " << f.first << "\n"
                     << "source:\n"
                     << f.second << '\n';
        inMemoryFs->addFile(f.first, 0, llvm::MemoryBuffer::getMemBuffer(f.second));
    }

    return overlayFs;
}
}  // namespace oklt
