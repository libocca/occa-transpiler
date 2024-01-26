#include <oklt/util/io_helper.h>

#include <clang/Basic/FileManager.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace oklt::util {

tl::expected<std::string, int> readFileAsStr(const std::filesystem::path &src_file) {
  clang::FileSystemOptions opts;
  clang::FileManager mng(opts);

  // read source code from file
  auto file_entry = mng.getFile(src_file.c_str());
  if (!file_entry) {
    return  tl::make_unexpected(file_entry.getError().value());
  }

  auto src = mng.getBufferForFile(file_entry.get());
  if (!src) {
    return  tl::make_unexpected(file_entry.getError().value());
  }
  auto src_str = src->get()->getBuffer();

  return std::string(src_str);
}

tl::expected<void, int> writeFileAsStr(const std::filesystem::path &srcPath,  std::string_view srcStr) {
  using namespace llvm;

  std::error_code errCode;
  llvm::raw_fd_ostream outFile(srcPath.string(), errCode);
  outFile.write(srcStr.data(), srcStr.size());
  outFile.flush();
  return {};
}

}
