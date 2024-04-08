#include "core/transpiler_session/original_source_mapper.h"
#include "core/rewriter/impl/dtree_rewriter_proxy.h"

#include <spdlog/spdlog.h>

#include <clang/AST/AST.h>

namespace oklt {
bool OriginalSourceMapper::addOriginalLine(uint32_t lineNumber, const std::string& line) {
    if (_originalLines.count(lineNumber) != 0) {
        return false;
    }
    _originalLines.insert(std::make_pair(lineNumber, line));

    return true;
}
bool OriginalSourceMapper::addAttributeColumn(clang::SourceLocation loc,
                                              uint32_t col,
                                              oklt::Rewriter& rewriter) {
    if (auto* dtreeRewriter = dynamic_cast<DtreeRewriterProxy*>(&rewriter)) {
        auto& dtrees = dtreeRewriter->getDeltaTrees();
        auto newOklAttrOffset = dtrees.getNewOffset(loc);
        auto fid = rewriter.getSourceMgr().getFileID(loc);
        _attrOffsetToOriginalCol[std::make_pair(fid, newOklAttrOffset)] = col;
        return true;
    }

    return false;
}

bool OriginalSourceMapper::updateAttributeColumns(oklt::Rewriter& rewriter) {
    if (auto* dtreeRewriter = dynamic_cast<DtreeRewriterProxy*>(&rewriter)) {
        auto& dtrees = dtreeRewriter->getDeltaTrees();
        AttributeColumns newAttrOffsetToOriginalCol;
        for (const auto [fidPrevNewOffset, col] : _attrOffsetToOriginalCol) {
            auto [fid, prevNewOffset] = fidPrevNewOffset;
            auto newOffset = dtrees.getNewOffset(fid, prevNewOffset);
            newAttrOffsetToOriginalCol[{fid, newOffset}] = col;
            SPDLOG_DEBUG("attribute offset: {}, original column: {}", newOffset, col);
        }
        _attrOffsetToOriginalCol = newAttrOffsetToOriginalCol;
        return true;
    }
    return false;
}

const OriginalLines& OriginalSourceMapper::getOriginalLines() {
    return _originalLines;
}

const AttributeColumns& OriginalSourceMapper::getAttrOffsetToOriginalCol() {
    return _attrOffsetToOriginalCol;
}
}  // namespace oklt
