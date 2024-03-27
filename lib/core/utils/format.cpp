#include <oklt/util/format.h>

#include <clang/Format/Format.h>
#include <clang/Tooling/Core/Replacement.h>

#include <vector>

#include <spdlog/spdlog.h>

using namespace clang;
using namespace clang::tooling;

namespace oklt {
std::string format(std::string_view code) {
    const std::vector<Range> ranges(1, Range(0, code.size()));
    auto style = format::getLLVMStyle();
    style.MaxEmptyLinesToKeep = 1;
    style.SeparateDefinitionBlocks = format::FormatStyle::SeparateDefinitionStyle::SDS_Always;

    Replacements replaces = format::reformat(style, code, ranges);
    auto changedCode = applyAllReplacements(code, replaces);
    if (!changedCode) {
        SPDLOG_ERROR("{}", toString(changedCode.takeError()));
        return {};
    }
    return changedCode.get();
}
}  // namespace oklt
