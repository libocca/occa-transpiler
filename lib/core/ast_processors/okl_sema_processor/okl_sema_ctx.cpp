#include <oklt/core/error.h>

#include "attributes/attribute_names.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/utils/ast_node_parsers.h"
#include "core/utils/type_converter.h"

#include <clang/AST/Attr.h>

namespace {
using namespace clang;
using namespace oklt;

DatatypeCategory toDatatypeCategory(const QualType& qt) {
    if (qt->isBuiltinType()) {
        return DatatypeCategory::BUILTIN;
    }
    return DatatypeCategory::CUSTOM;
}

bool hasParentLoopConflictWithNestedLoop(std::string_view parentAttr, std::string_view childAttr) {
    static std::map<std::string_view, uint32_t> loopScore = {
        {OUTER_ATTR_NAME, 1}, {TILE_ATTR_NAME, 1}, {INNER_ATTR_NAME, 0}};

    if (loopScore[parentAttr] < loopScore[childAttr]) {
        return true;
    }

    return false;
}

using OklForStmt = OklSemaCtx::ParsingKernelInfo::OklForStmt;
tl::expected<OklForStmt, Error> makeOklForStmt(const clang::ForStmt& forStmt,
                                               const clang::Attr& attr,
                                               ASTContext& ctx) {
    auto parsedLoopMeta = parseForStmt(forStmt, ctx);
    if (!parsedLoopMeta) {
        return tl::make_unexpected(std::move(parsedLoopMeta.error()));
    }

    return OklForStmt{.attr = attr, .stmt = &forStmt, .meta = std::move(parsedLoopMeta.value())};
}

using ParsedLoopBlock = OklSemaCtx::ParsingKernelInfo::ParsedLoopBlock;
tl::expected<ParsedLoopBlock, Error> makeParsedLoopBlock(const clang::ForStmt& stmt,
                                                         const clang::Attr& attr,
                                                         ASTContext& ctx) {
    return makeOklForStmt(stmt, attr, ctx)
        .and_then([&](auto&& val) -> tl::expected<ParsedLoopBlock, Error> {
            return ParsedLoopBlock{.loopLocs = stmt.getBeginLoc(),
                                   .numInnerThreads = 0,
                                   .nestedLoops = {std::move(val)}};
        });
}

bool isLegalTopLevelLoopAttr(std::string_view attrName) {
    return (attrName == TILE_ATTR_NAME || attrName == OUTER_ATTR_NAME);
}
}  // namespace

namespace oklt {
OklSemaCtx::ParsingKernelInfo* OklSemaCtx::startParsingOklKernel(const FunctionDecl& fd) {
    if (_parsingKernInfo.has_value()) {
        return nullptr;
    }

    auto* kiPtr = &_programMetaData.addKernelInfo(fd.getNameAsString(), fd.param_size(), 1);

    // link created slot with current parsing kernel context
    _parsingKernInfo = ParsingKernelInfo{.kernInfo = kiPtr,
                                         .argStrs = std::vector<std::string>(fd.param_size()),
                                         .kernFuncDecl = &fd};

    return &_parsingKernInfo.value();
}

OklSemaCtx::ParsingKernelInfo* OklSemaCtx::getParsingKernelInfo() {
    return _parsingKernInfo ? &_parsingKernInfo.value() : nullptr;
}

void OklSemaCtx::stopParsingKernelInfo() {
    _parsingKernInfo.reset();
}

bool OklSemaCtx::isParsingOklKernel() const {
    return _parsingKernInfo.has_value();
}

bool OklSemaCtx::isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const {
    if (!_parsingKernInfo) {
        return false;
    }

    return _parsingKernInfo->kernFuncDecl == &fd;
}

bool OklSemaCtx::isDeclInLexicalTraversal(const Decl& decl) const {
    if (!_parsingKernInfo.has_value()) {
        return false;
    }

    return cast<FunctionDecl>(decl.getParentFunctionOrMethod()) == _parsingKernInfo->kernFuncDecl;
}

std::optional<LoopMetaData> OklSemaCtx::getLoopMetaData(const clang::ForStmt& forStmt) const {
    if (!_parsingKernInfo) {
        return std::nullopt;
    }

    if (_parsingKernInfo->state == ParsingKernelInfo::LoopBlockParserState::NotStarted) {
        return std::nullopt;
    }

    auto& loops = _parsingKernInfo->parsingLoopBlockIt->nestedLoops;
    auto it = std::find_if(
        loops.begin(), loops.end(), [&forStmt](const auto& l) { return l.stmt == &forStmt; });
    if (it == loops.end()) {
        return std::nullopt;
    }

    return it->meta;
}

tl::expected<void, Error> OklSemaCtx::validateOklForLoopOnPreTraverse(const clang::Attr& attr,
                                                                      const clang::ForStmt& stmt) {
    switch (_parsingKernInfo->state) {
        case ParsingKernelInfo::LoopBlockParserState::NotStarted: {
            if (!isLegalTopLevelLoopAttr(attr.getNormalizedFullName())) {
                // TODO add source loc
                return tl::make_unexpected(Error{
                    .ec = std::error_code(), .desc = "first loop is not outer/tile attributed"});
            }
            // make loop block and set iterator cursor to it
            return makeParsedLoopBlock(stmt, attr, _parsingKernInfo->kernFuncDecl->getASTContext())
                .and_then([this](auto&& loopBlock) -> tl::expected<void, Error> {
                    _parsingKernInfo->outerLoopBlocks.emplace_back(std::move(loopBlock));
                    _parsingKernInfo->parsingLoopBlockIt =
                        std::prev(_parsingKernInfo->outerLoopBlocks.end());
                    _parsingKernInfo->state = ParsingKernelInfo::LoopBlockParserState::PreTraverse;
                    return {};
                });
        }
        case ParsingKernelInfo::LoopBlockParserState::PreTraverse: {
            const auto& parentLoop = _parsingKernInfo->parsingLoopBlockIt->nestedLoops.back()
                                         .attr.getNormalizedFullName();
            if (hasParentLoopConflictWithNestedLoop(parentLoop, attr.getNormalizedFullName())) {
                // TODO add source loc
                return tl::make_unexpected(
                    Error{.ec = std::error_code(), .desc = "tile/outer after inner"});
            }

            // push nested loop
            return makeOklForStmt(stmt, attr, _parsingKernInfo->kernFuncDecl->getASTContext())
                .and_then([this](auto&& val) -> tl::expected<void, Error> {
                    _parsingKernInfo->parsingLoopBlockIt->nestedLoops.push_back(std::move(val));
                    return {};
                });
        }
        case ParsingKernelInfo::LoopBlockParserState::PostTraverse: {
            // TODO add source loc
            return tl::make_unexpected(
                Error{.ec = std::error_code(), .desc = "push loop on post traverse"});
        }
    }

    // TODO add source loc
    return tl::make_unexpected(Error{.ec = std::error_code(), .desc = "buggy"});
}

tl::expected<void, Error> OklSemaCtx::validateOklForLoopOnPostTraverse(const clang::Attr& attr,
                                                                       const clang::ForStmt& stmt) {
    assert(_parsingKernInfo);

    switch (_parsingKernInfo->state) {
        case ParsingKernelInfo::PreTraverse: {
            auto& loops = _parsingKernInfo->parsingLoopBlockIt->nestedLoops;
            // set iterator for most nested loop and validate sema
            _parsingKernInfo->postLoopIt = std::prev(loops.end());

            // single tile loop, fill output metadata, reset state and go on
            if (loops.size() == 1) {
                if (_parsingKernInfo->postLoopIt->attr.getNormalizedFullName() != TILE_ATTR_NAME) {
                    // TODO add source loc
                    return tl::make_unexpected(Error{.ec = std::error_code(),
                                                     .desc = "single loop is not tile attributed"});
                }
                _parsingKernInfo->state = ParsingKernelInfo::NotStarted;
                return {};
            }

            --_parsingKernInfo->postLoopIt;
            _parsingKernInfo->state = ParsingKernelInfo::PostTraverse;
        } break;
        case ParsingKernelInfo::PostTraverse: {
            // all loops in block are processed - reset state and go on
            if (_parsingKernInfo->postLoopIt ==
                _parsingKernInfo->parsingLoopBlockIt->nestedLoops.begin()) {
                _parsingKernInfo->state = ParsingKernelInfo::NotStarted;
                return {};
            }
            --_parsingKernInfo->postLoopIt;
            _parsingKernInfo->state = ParsingKernelInfo::PostTraverse;
        } break;
        case ParsingKernelInfo::LoopBlockParserState::NotStarted: {
            // TODO add source loc
            return tl::make_unexpected(
                Error{.ec = std::error_code(), .desc = "OKL loop block is not parsing"});
        }
    }

    return {};
}

void OklSemaCtx::setKernelArgInfo(const ParmVarDecl& parm) {
    assert(_parsingKernInfo.has_value());
    auto result = toOklArgInfo(parm);
    if (!result) {
        llvm::errs() << "failed to convert parm var decl to okl data type\n";
        return;
    }

    auto* ki = _parsingKernInfo.value().kernInfo;
    ki->args[parm.getFunctionScopeIndex()] = std::move(result.value());
}

void OklSemaCtx::setKernelArgRawString(const ParmVarDecl& parm, std::string_view transpiledType) {
    assert(_parsingKernInfo.has_value());

    auto varType = [](const auto& p, auto transpiledType) {
        return !transpiledType.empty() ? std::string(transpiledType) : p.getType().getAsString();
    }(parm, transpiledType);

    auto& pki = _parsingKernInfo.value();
    pki.argStrs[parm.getFunctionScopeIndex()] = varType + " " + parm.getNameAsString();
}

void OklSemaCtx::setKernelTranspiledAttrStr(std::string attrStr) {
    assert(_parsingKernInfo.has_value());
    _parsingKernInfo.value().transpiledFuncAttrStr = std::move(attrStr);
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
    return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
    return _programMetaData;
}
}  // namespace oklt
