#pragma once

#include <oklt/core/metadata/program.h>

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
        std::stack<const clang::ForStmt*> loopStack;
        std::stack<std::string_view> loopAttrStack;
    };

    OklSemaCtx() = default;

    // method to make/get/reset context of parsing OKL kernel
    ParsingKernelInfo* startParsingOklKernel(const clang::FunctionDecl*);
    [[nodiscard]] ParsingKernelInfo* getParsingKernelInfo();
    void stopParsingKernelInfo();

    [[nodiscard]] bool isParsingOklKernel() const;
    [[nodiscard]] bool isKernelParmVar(const clang::ParmVarDecl*) const;

    [[nodiscard]] tl::expected<bool, Error> validateForSeries(const clang::ForStmt*,
                                                              const clang::Attr*);

    void setKernelArgInfo(const clang::ParmVarDecl* parm);
    void setKernelArgRawString(const clang::ParmVarDecl* parm,
                               std::string_view transpiledType = {});

    void setKernelTranspiledAttrStr(std::string attrStr);

    [[nodiscard]] ProgramMetaData& getProgramMetaData();
    [[nodiscard]] const ProgramMetaData& getProgramMetaData() const;

   private:
    std::optional<ParsingKernelInfo> _parsingKernInfo;
    ProgramMetaData _programMetaData;
};
}  // namespace oklt
