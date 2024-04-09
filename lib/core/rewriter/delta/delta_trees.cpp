#include "core/rewriter/delta/delta_trees.h"

#include <spdlog/spdlog.h>

#include <clang/Lex/Lexer.h>

DeltaTrees::DeltaTrees(clang::CompilerInstance& compiler)
    : _SM(&compiler.getSourceManager()),
      _LO(&compiler.getLangOpts()) {}

DeltaTrees::DeltaTrees(const clang::SourceManager& SM, const clang::LangOptions& LO)
    : _SM(&SM),
      _LO(&LO) {}

bool DeltaTrees::Remove(clang::SourceLocation loc, size_t size) {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");

    auto [fid, startOffset] = _SM->getDecomposedLoc(loc);
    auto& dtree = getTree(fid);

    AddReplaceDelta(dtree, startOffset, -size);
    return true;
}

bool DeltaTrees::Remove(clang::SourceRange range) {
    return Remove(range.getBegin(), getRangeSize(range));
}

bool DeltaTrees::Insert(clang::SourceLocation loc, size_t size, bool InsertAfter) {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");

    auto [fid, offset] = _SM->getDecomposedLoc(loc);
    auto& dtree = getTree(fid);
    AddInsertDelta(dtree, offset, size);
    return true;
}
bool DeltaTrees::Replace(clang::SourceLocation loc, size_t oldSize, size_t newSIze) {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");

    auto [fid, offset] = _SM->getDecomposedLoc(loc);

    auto& dtree = getTree(fid);
    if (oldSize != newSIze) {
        AddReplaceDelta(dtree, offset, static_cast<int>(newSIze) - static_cast<int>(oldSize));
    }
    return true;
}

unsigned DeltaTrees::getNewOffset(clang::SourceLocation loc, bool afterInserts) {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");

    auto [fid, offset] = _SM->getDecomposedLoc(loc);
    auto& dtree = getTree(fid);
    return getMappedOffset(dtree, offset, afterInserts);
}

unsigned DeltaTrees::getNewOffset(clang::SourceLocation loc, bool afterInserts) const {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");

    auto [fid, offset] = _SM->getDecomposedLoc(loc);

    auto dtreeIt = _dtrees.find(fid);
    if (dtreeIt == _dtrees.end()) {
        SPDLOG_DEBUG("DeltaTrees::getNewOffset called for unknown file: {}", fid.getHashValue());
        return offset;
    }

    return getMappedOffset(dtreeIt->second, offset, afterInserts);
}

unsigned DeltaTrees::getNewOffset(clang::FileID fid, uint32_t offset, bool afterInserts) {
    auto& dtree = getTree(fid);
    return getMappedOffset(dtree, offset, afterInserts);
}

unsigned DeltaTrees::getNewOffset(clang::FileID fid, uint32_t offset, bool afterInserts) const {
    auto dtreeIt = _dtrees.find(fid);
    if (dtreeIt == _dtrees.end()) {
        SPDLOG_DEBUG("DeltaTrees::getNewOffset called for unknown file: {}", fid.getHashValue());
        return offset;
    }

    return getMappedOffset(dtreeIt->second, offset, afterInserts);
}

clang::DeltaTree& DeltaTrees::getTree(clang::FileID fid) {
    auto [it, _] = _dtrees.try_emplace(fid, clang::DeltaTree{});
    return it->second;
}

unsigned DeltaTrees::getMappedOffset(const clang::DeltaTree& dtree,
                                     unsigned OrigOffset,
                                     bool AfterInserts) const {
    return dtree.getDeltaAt(2 * OrigOffset + AfterInserts) + OrigOffset;
}

/// AddInsertDelta - When an insertion is made at a position, this
/// method is used to record that information.
void DeltaTrees::AddInsertDelta(clang::DeltaTree& dtree, unsigned OrigOffset, int Change) {
    return dtree.AddDelta(2 * OrigOffset, Change);
}

/// AddReplaceDelta - When a replacement/deletion is made at a position, this
/// method is used to record that information.
void DeltaTrees::AddReplaceDelta(clang::DeltaTree& dtree, unsigned OrigOffset, int Change) {
    return dtree.AddDelta(2 * OrigOffset + 1, Change);
}

int DeltaTrees::getRangeSize(clang::SourceRange range) const {
    assert(_SM != nullptr && "DeltaTrees::_SM is nullptr");
    assert(_LO != nullptr && "DeltaTrees::_LO is nullptr");
    clang::CharSourceRange cRange = clang::CharSourceRange::getTokenRange(range);

    auto [StartFileID, StartOff] = _SM->getDecomposedLoc(cRange.getBegin());
    auto [EndFileID, EndOff] = _SM->getDecomposedLoc(cRange.getEnd());

    if (StartFileID != EndFileID) {
        return -1;
    }

    // If edits have been made to this buffer, the delta between the range may
    // have changed.
    auto it = _dtrees.find(StartFileID);
    if (it != _dtrees.end()) {
        const auto& dtree = it->second;
        EndOff = getMappedOffset(dtree, EndOff, true);
        StartOff = getMappedOffset(dtree, StartOff, false);
    }

    // Adjust the end offset to the end of the last token, instead of being the
    // start of the last token if this is a token range.
    if (cRange.isTokenRange()) {
        EndOff += clang::Lexer::MeasureTokenLength(cRange.getEnd(), *_SM, *_LO);
    }

    auto res = EndOff - StartOff;
    return res;
}
