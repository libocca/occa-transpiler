#pragma once

#include "core/sema/okl_sema_info.h"
#include "oklt/core/kernel_metadata.h"

#include <clang/AST/AST.h>

#include <tl/expected.hpp>

namespace oklt {

struct Error;

struct OklSemaCtx {
    struct ParsedKernelInfo : public OklKernelInfo {
        explicit ParsedKernelInfo(const clang::FunctionDecl& d,
                                  std::vector<std::string>&& args,
                                  KernelInfo* info = nullptr)
            : OklKernelInfo(d),
              argStrs(args),
              kernInfo(info){};
        KernelInfo* kernInfo{nullptr};
        std::list<OklLoopInfo> highestLevelLoops;

        std::string transpiledFuncAttrStr = {};
        std::vector<std::string> argStrs = {};

        OklLoopInfo* currentLoop = nullptr;
        std::map<const clang::ForStmt*, OklLoopInfo*> loopMap = {};
    };

    OklSemaCtx() = default;

    // method to make/get/reset context of parsing OKL kernel
    bool startParsingOklKernel(const clang::FunctionDecl&);
    void stopParsingKernelInfo();
    [[nodiscard]] ParsedKernelInfo* getParsingKernelInfo();
    void setParsedKernelInfo(ParsedKernelInfo*);

    [[nodiscard]] bool isParsingOklKernel() const;
    [[nodiscard]] bool isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const;
    [[nodiscard]] bool isDeclInLexicalTraversal(const clang::Decl&) const;

    [[nodiscard]] tl::expected<void, Error> startParsingAttributedForLoop(
        const clang::Attr& attr,
        const clang::ForStmt& stmt,
        const std::any* params);
    [[nodiscard]] tl::expected<void, Error> stopParsingAttributedForLoop(const clang::Attr& attr,
                                                                         const clang::ForStmt& stmt,
                                                                         const std::any* params);
    [[nodiscard]] OklLoopInfo* getLoopInfo(const clang::ForStmt& forStmt) const;
    [[nodiscard]] OklLoopInfo* getLoopInfo();
    void setLoopInfo(OklLoopInfo* loopInfo);

    void setKernelArgInfo(const clang::ParmVarDecl& parm);
    void setTranspiledArgStr(const clang::ParmVarDecl& parm,
                             std::string_view transpiledArgStr = {});

    void setKernelTranspiledAttrStr(std::string attrStr);

    [[nodiscard]] ProgramMetaData& getProgramMetaData();
    [[nodiscard]] const ProgramMetaData& getProgramMetaData() const;

   private:
    ParsedKernelInfo* _parsingKernInfo = nullptr;
    std::list<ParsedKernelInfo> _parsedKernelList;
    ProgramMetaData _programMetaData;
};
}  // namespace oklt
