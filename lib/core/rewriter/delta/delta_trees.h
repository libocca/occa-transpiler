#pragma once

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Rewrite/Core/DeltaTree.h>
#include <map>

class DeltaTrees {
    std::map<clang::FileID, clang::DeltaTree> _dtrees;
    const clang::SourceManager* _SM;
    const clang::LangOptions* _LO;

   public:
    DeltaTrees() = default;
    DeltaTrees(clang::CompilerInstance& compiler);
    DeltaTrees(const clang::SourceManager& SM, const clang::LangOptions& LO);
    ~DeltaTrees() { int x; }

    bool Remove(clang::SourceLocation loc, size_t size);
    bool Remove(clang::SourceRange range);

    bool Insert(clang::SourceLocation loc, size_t size, bool InsertAfter);

    bool Replace(clang::SourceLocation loc, size_t oldSize, size_t newSIze);

    unsigned getNewOffset(clang::SourceLocation loc, bool afterInserts = false);
    unsigned getNewOffset(clang::SourceLocation loc, bool afterInserts = false) const;

    unsigned getNewOffset(clang::FileID fid, uint32_t offset, bool afterInserts = false);
    unsigned getNewOffset(clang::FileID fid, uint32_t offset, bool afterInserts = false) const;

    int getRangeSize(clang::SourceRange range) const;

   private:
    clang::DeltaTree& getTree(clang::FileID fid);

    unsigned getMappedOffset(const clang::DeltaTree& dtree,
                             unsigned OrigOffset,
                             bool AfterInserts = false) const;

    void AddInsertDelta(clang::DeltaTree& dtree, unsigned OrigOffset, int Change);

    void AddReplaceDelta(clang::DeltaTree& dtree, unsigned OrigOffset, int Change);
};
