#include "oklt/util/file_helper.h"
#include <clang/Basic/FileManager.h>

namespace oklt::util {

llvm::Expected<std::string> read_file_as_str(const std::filesystem::path &src_file) {
  clang::FileSystemOptions opts;
  clang::FileManager mng(opts);

  // read source code from file
  auto file_entry = mng.getFile(src_file.c_str());
  if (!file_entry) {
    return llvm::createStringError(file_entry.getError(), file_entry.getError().message());
  }

  auto src = mng.getBufferForFile(file_entry.get());
  if (!src) {
    return llvm::createStringError(src.getError(), src.getError().message());
  }
  auto src_str = src->get()->getBuffer();

  return std::string(src_str);
}

llvm::Error write_file_as_str(const std::filesystem::path &src_path,  std::string_view src_str) {
  clang::FileSystemOptions opts;
  clang::FileManager mng(opts);

  auto file_ref = mng.getFileRef(src_path.generic_string());
  if (!file_ref) {
    return file_ref.takeError();
  }

  return llvm::Error::success();
}

}
