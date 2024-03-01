#pragma once

#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <optional>
#include <stack>

#include <tl/expected.hpp>

namespace oklt {

struct KernelInfo;
struct Error;

struct OklSemaCtx {
    struct ParsingKernelInfo {
        KernelInfo* kernInfo{nullptr};

        std::string transpiledFuncAttrStr;
        std::vector<std::string> argStrs;

        const clang::FunctionDecl* kernFuncDecl;
        std::stack<const clang::CompoundStmt*> compoundStack;

        enum LoopBlockParserState { NotStarted, PreTraverse, PostTraverse };
        LoopBlockParserState state{NotStarted};

        struct OklForStmt {
            const clang::Attr& attr;
            const clang::ForStmt* stmt;
            LoopMetaData meta;
        };
        struct ParsedLoopBlock {
            clang::SourceLocation loopLocs;
            uint32_t numInnerThreads;
            std::list<OklForStmt> nestedLoops;
        };
        std::list<ParsedLoopBlock> outerLoopBlocks;
        std::list<ParsedLoopBlock>::iterator parsingLoopBlockIt;
        std::list<OklForStmt>::iterator postLoopIt;
    };

    OklSemaCtx() = default;

    // method to make/get/reset context of parsing OKL kernel
    ParsingKernelInfo* startParsingOklKernel(const clang::FunctionDecl&);
    [[nodiscard]] ParsingKernelInfo* getParsingKernelInfo();
    void stopParsingKernelInfo();

    [[nodiscard]] bool isParsingOklKernel() const;
    [[nodiscard]] bool isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const;
    [[nodiscard]] bool isDeclInLexicalTraversal(const clang::Decl&) const;

    [[nodiscard]] std::optional<LoopMetaData> getLoopMetaData(const clang::ForStmt& forStmt) const;

    [[nodiscard]] tl::expected<void, Error> validateOklForLoopOnPreTraverse(const clang::Attr&,
                                                                            const clang::ForStmt&);
    [[nodiscard]] tl::expected<void, Error> validateOklForLoopOnPostTraverse(const clang::Attr&,
                                                                             const clang::ForStmt&);

    void setKernelArgInfo(const clang::ParmVarDecl& parm);
    void setTranspiledArgStr(const clang::ParmVarDecl& parm,
                             std::string_view transpiledArgStr = {});

    void setKernelTranspiledAttrStr(std::string attrStr);

    [[nodiscard]] ProgramMetaData& getProgramMetaData();
    [[nodiscard]] const ProgramMetaData& getProgramMetaData() const;

   private:
    std::optional<ParsingKernelInfo> _parsingKernInfo;
    ProgramMetaData _programMetaData;
};
}  // namespace oklt
