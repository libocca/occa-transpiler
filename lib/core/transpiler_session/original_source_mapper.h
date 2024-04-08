#pragma once

#include <cstdint>
#include <map>
#include <string>
#include "core/rewriter/rewriter_proxy.h"

namespace clang {
class FileID;
class SourceLocation;
}  // namespace clang

namespace oklt {
// Modified line number -> Original source line
using OriginalLines = std::map<uint32_t, std::string>;
// New File, attribute begin offset -> column in the original line
using AttributeColumns = std::map<std::pair<clang::FileID, uint32_t>, uint32_t>;

class OriginalSourceMapper {
   public:
    OriginalSourceMapper() = default;

    bool addOriginalLine(uint32_t lineNumber, const std::string& line);
    bool addAttributeColumn(clang::SourceLocation loc, uint32_t col, oklt::Rewriter& rewriter);
    bool updateAttributeColumns(oklt::Rewriter& rewriter);

    const OriginalLines& getOriginalLines();
    const AttributeColumns& getAttrOffsetToOriginalCol();

   private:
    OriginalLines _originalLines;
    AttributeColumns _attrOffsetToOriginalCol;
};
}  // namespace oklt
