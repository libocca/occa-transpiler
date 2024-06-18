#include "core/vfs/overlay_fs.h"

namespace oklt {
using namespace llvm;
IntrusiveRefCntPtr<vfs::FileSystem> makeOverlayFs(IntrusiveRefCntPtr<vfs::FileSystem> baseFs,
                                                  const std::map<std::string, std::string>& files) {
    IntrusiveRefCntPtr<vfs::OverlayFileSystem> overlayFs(
        new vfs::OverlayFileSystem(vfs::getRealFileSystem()));
    IntrusiveRefCntPtr<vfs::InMemoryFileSystem> inMemoryFs(new vfs::InMemoryFileSystem);

    //  ovelay is FS stack - ORDER MATTER
    overlayFs->pushOverlay(baseFs);
    overlayFs->pushOverlay(inMemoryFs);

    for (const auto& f : files) {
        inMemoryFs->addFile(f.first, 0, MemoryBuffer::getMemBuffer(f.second));
    }

    return overlayFs;
}
}  // namespace oklt
