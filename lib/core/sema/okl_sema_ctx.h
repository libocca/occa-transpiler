#pragma once

#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <optional>

#include <tl/expected.hpp>

namespace oklt {

struct KernelInfo;
struct Error;

struct OklLoopInfo {
    const clang::Attr& attr;
    const clang::ForStmt& stmt;
    LoopMetaData& metadata;

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};

    // TODO: Maybe make it via `Extendable` class ?
    struct {
        std::list<std::reference_wrapper<const clang::Decl>> shared = {};
        std::list<std::reference_wrapper<const clang::VarDecl>> exclusive = {};
    } vars;

    OklLoopInfo* getAttributedParent();
    OklLoopInfo* getFirstAttributedChild();
    std::optional<size_t> getSize();
};

struct OklKernelInfo {
    explicit OklKernelInfo(const clang::FunctionDecl& decl)
        : decl(std::ref(decl)){};
    const std::reference_wrapper<const clang::FunctionDecl> decl;
    std::list<OklLoopInfo> children = {};
};

struct OklSemaCtx {
    struct ParsedKernelInfo : public OklKernelInfo {
        explicit ParsedKernelInfo(const clang::FunctionDecl& d,
                                  std::vector<std::string>&& args,
                                  KernelInfo* info = nullptr)
            : OklKernelInfo(d),
              argStrs(args),
              kernInfo(info){};
        KernelInfo* kernInfo{nullptr};

        std::string transpiledFuncAttrStr = {};
        std::vector<std::string> argStrs = {};

        OklLoopInfo* currentLoop = nullptr;
        std::map<const clang::ForStmt*, OklLoopInfo*> loopMap = {};
    };

    OklSemaCtx() = default;

    // method to make/get/reset context of parsing OKL kernel
    bool startParsingOklKernel(const clang::FunctionDecl&);
    [[nodiscard]] ParsedKernelInfo* getParsingKernelInfo();
    void setParsedKernelInfo(ParsedKernelInfo*);
    void stopParsingKernelInfo();

    [[nodiscard]] bool isParsingOklKernel() const;
    [[nodiscard]] bool isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const;
    [[nodiscard]] bool isDeclInLexicalTraversal(const clang::Decl&) const;

    [[nodiscard]] OklLoopInfo* getLoopInfo(const clang::ForStmt& forStmt) const;
    [[nodiscard]] OklLoopInfo* getLoopInfo();
    void setLoopInfo(OklLoopInfo* loopInfo);

    [[nodiscard]] tl::expected<void, Error> validateOklForLoopOnPreTraverse(const clang::Attr&,
                                                                            const clang::ForStmt&,
                                                                            const std::any* params);
    [[nodiscard]] tl::expected<void, Error> validateOklForLoopOnPostTraverse(
        const clang::Attr&,
        const clang::ForStmt&,
        const std::any* params);

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
