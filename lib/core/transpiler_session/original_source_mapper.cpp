#include "core/transpiler_session/original_source_mapper.h"
#include "core/rewriter/impl/dtree_rewriter_proxy.h"

#include <spdlog/spdlog.h>

#include <clang/AST/AST.h>

namespace oklt {
bool OriginalSourceMapper::addOriginalLine(FiDUintPair fidLineNumber, const std::string& line) {
    if (_originalLines.count(fidLineNumber) != 0) {
        return false;
    }
    _originalLines.insert(std::make_pair(fidLineNumber, line));

    return true;
}
bool OriginalSourceMapper::addAttributeColumn(clang::SourceLocation loc,
                                              uint32_t col,
                                              oklt::Rewriter& rewriter,
                                              uint32_t addOffset) {
    if (auto* dtreeRewriter = dynamic_cast<DtreeRewriterProxy*>(&rewriter)) {
        auto& dtrees = dtreeRewriter->getDeltaTrees();
        auto newOklAttrOffset = dtrees.getNewOffset(loc) + addOffset;
        auto fid = rewriter.getSourceMgr().getFileID(loc);
        _attrOffsetToOriginalCol[std::make_pair(fid, newOklAttrOffset)] = col;
        SPDLOG_DEBUG("Insert (fid: {} attr offset: {}) -> col: {}",
                     fid.getHashValue(),
                     newOklAttrOffset,
                     col);
        return true;
    }

    return false;
}

bool OriginalSourceMapper::updateAttributeOffset(FiDUintPair prevFidOffset,
                                                 clang::SourceLocation newLoc,
                                                 oklt::Rewriter& rewriter,
                                                 uint32_t addOffset) {
    if (auto* dtreeRewriter = dynamic_cast<DtreeRewriterProxy*>(&rewriter)) {
        auto& dtrees = dtreeRewriter->getDeltaTrees();
        if (_attrOffsetToOriginalCol.find(prevFidOffset) == _attrOffsetToOriginalCol.end()) {
            return false;
        }

        auto col = _attrOffsetToOriginalCol.at(prevFidOffset);
        _attrOffsetToOriginalCol.erase(prevFidOffset);
        auto newOffset = dtrees.getNewOffset(newLoc) + addOffset;
        auto fid = prevFidOffset.first;
        _attrOffsetToOriginalCol[{fid, newOffset}] = col;
        SPDLOG_DEBUG("Update fid: {}, attribute offset {} -> {}, col: {}",
                     fid.getHashValue(),
                     prevFidOffset.second,
                     newOffset,
                     col);

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
