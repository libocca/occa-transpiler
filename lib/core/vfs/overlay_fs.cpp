#include "core/vfs/overlay_fs.h"
#include "core/transpiler_session/header_info.h"

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
        std::time_t current_time = std::time(0);
        inMemoryFs->addFile(f.first, current_time, MemoryBuffer::getMemBuffer(f.second));
    }

    return overlayFs;
}
}  // namespace oklt
