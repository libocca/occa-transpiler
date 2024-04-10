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
using FiDUintPair = std::pair<clang::FileID, uint32_t>;
// Modified line number -> Original source line
using OriginalLines = std::map<FiDUintPair, std::string>;
// New File, attribute begin offset -> column in the original line
using AttributeColumns = std::map<FiDUintPair, uint32_t>;

class OriginalSourceMapper {
   public:
    OriginalSourceMapper() = default;

    bool addOriginalLine(FiDUintPair fidLineNumber, const std::string& line);
    bool addAttributeColumn(clang::SourceLocation loc,
                            uint32_t col,
                            oklt::Rewriter& rewriter,
                            uint32_t addOffset = 0);
    bool updateAttributeOffset(FiDUintPair prevFidOffset,
                               clang::SourceLocation newLoc,
                               oklt::Rewriter& rewriter,
                               uint32_t addOffset = 0);

    const OriginalLines& getOriginalLines();
    const AttributeColumns& getAttrOffsetToOriginalCol();

   private:
    OriginalLines _originalLines;
    AttributeColumns _attrOffsetToOriginalCol;
};
}  // namespace oklt
