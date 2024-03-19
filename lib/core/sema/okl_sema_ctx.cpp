#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"

#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/ast_node_parsers.h"
#include "core/utils/type_converter.h"
#include "oklt/core/kernel_metadata.h"

#include <oklt/core/error.h>
#include <oklt/util/string_utils.h>

#include <clang/AST/Attr.h>

namespace {
using namespace clang;
using namespace oklt;

AttributedLoopTypes getLoopType(const std::any* param) {
    if (!param) {
        return {LoopType::Regular};
    }

    AttributedLoopTypes res{};
    if (param->type() == typeid(TileParams)) {
        auto tile = std::any_cast<TileParams>(*param);
        res.push_back(tile.firstLoop.type);
        res.push_back(tile.secondLoop.type);
    } else if (param->type() == typeid(AttributedLoop)) {
        auto loop = std::any_cast<AttributedLoop>(*param);
        res.push_back(LoopType{loop.type});
    }

    return res;
}

tl::expected<OklLoopInfo, Error> makeOklLoopInfo(const clang::ForStmt& stmt,
                                                 const clang::Attr& attr,
                                                 AttributedLoopTypes loopType,
                                                 OklSemaCtx::ParsedKernelInfo& kernelInfo,
                                                 SessionStage& stage) {
    auto parsedLoopInfo = parseForStmt(attr, stmt, stage);
    if (!parsedLoopInfo) {
        return parsedLoopInfo;
    }
    parsedLoopInfo->type = loopType;
    return parsedLoopInfo;
}

// check if loop types inside one loop are legal. firstType/lastType - first and alst non regular
// loop type
bool isLegalLoopLevel(AttributedLoopTypes loopTypes,
                      LoopType& firstType,
                      LoopType& lastType) {
    lastType = LoopType::Regular;
    firstType = LoopType::Regular;
    for (auto& loopType : loopTypes) {
        if (loopType != LoopType::Regular && firstType == LoopType::Regular) {
            firstType = loopType;
        }
        if (loopType == LoopType::Inner) {
            lastType = LoopType::Inner;
        } else if (loopType == LoopType::Outer) {
            if (lastType == LoopType::Inner) {
                // inner -> outer inside parent type
                return false;
            }
            lastType = LoopType::Outer;
        }
    }
    return true;
}

bool isLegalLoopLevel(AttributedLoopTypes childTypes,
                      AttributedLoopTypes parentTypes = {LoopType::Regular}) {
    LoopType firstParentType = LoopType::Regular,
                       lastParentType = LoopType::Regular;
    LoopType firstChildType = LoopType::Regular,
                       lastChildType = LoopType::Regular;
    if (!isLegalLoopLevel(parentTypes, firstParentType, lastParentType)) {
        return false;
    }
    if (!isLegalLoopLevel(childTypes, firstChildType, lastChildType)) {
        return false;
    }

    if (lastParentType == firstChildType) {
        return true;
    }

    if (firstChildType == LoopType::Regular ||
        lastParentType == LoopType::Regular) {
        return true;
    }

    if (lastParentType == LoopType::Outer &&
        firstChildType == LoopType::Inner) {
        return true;
    }

    return false;
}

bool isLegalTopLoopLevel(AttributedLoopTypes loopType) {
    return loopType.front() == LoopType::Outer;
}

}  // namespace

namespace oklt {

void OklSemaCtx::clear() {
    _parsingKernInfo = nullptr;
    _parsedKernelList.clear();
    _programMetaData = ProgramMetaData();
}

bool OklSemaCtx::startParsingOklKernel(const FunctionDecl& fd) {
    if (_parsingKernInfo) {
        return false;
    }

    _parsingKernInfo = &_parsedKernelList.emplace_back(fd);
    return true;
}

OklSemaCtx::ParsedKernelInfo* OklSemaCtx::getParsingKernelInfo() {
    return _parsingKernInfo;
}

void OklSemaCtx::setParsedKernelInfo(ParsedKernelInfo* ki) {
    _parsingKernInfo = ki;
}

void OklSemaCtx::stopParsingKernelInfo() {
    _parsingKernInfo = nullptr;
}

bool OklSemaCtx::isParsingOklKernel() const {
    return _parsingKernInfo;
}

bool OklSemaCtx::isCurrentParsingOklKernel(const clang::FunctionDecl& fd) const {
    if (!_parsingKernInfo) {
        return false;
    }

    return std::addressof(_parsingKernInfo->decl.get()) == std::addressof(fd);
}

bool OklSemaCtx::isDeclInLexicalTraversal(const Decl& decl) const {
    if (!_parsingKernInfo) {
        return false;
    }

    return cast<FunctionDecl>(decl.getParentFunctionOrMethod()) ==
           std::addressof(_parsingKernInfo->decl.get());
}

OklLoopInfo* OklSemaCtx::getLoopInfo(const clang::ForStmt& forStmt) const {
    if (!_parsingKernInfo) {
        return nullptr;
    }

    auto it = _parsingKernInfo->loopMap.find(&forStmt);
    if (it == _parsingKernInfo->loopMap.end() || !it->second) {
        return nullptr;
    }

    return it->second;
}

[[nodiscard]] OklLoopInfo* OklSemaCtx::getLoopInfo() {
    if (!_parsingKernInfo) {
        return nullptr;
    }

    return _parsingKernInfo->currentLoop;
}

void OklSemaCtx::setLoopInfo(OklLoopInfo* loopInfo) {
    if (!_parsingKernInfo) {
        return;
    }

    auto it = std::find_if(_parsingKernInfo->loopMap.begin(),
                           _parsingKernInfo->loopMap.end(),
                           [loopInfo](const auto& v) { return v.second == loopInfo; });
    if (it != _parsingKernInfo->loopMap.end()) {
        _parsingKernInfo->currentLoop = loopInfo;
    }
}

tl::expected<void, Error> OklSemaCtx::startParsingAttributedForLoop(const clang::Attr& attr,
                                                                    const clang::ForStmt& stmt,
                                                                    const std::any* params,
                                                                    SessionStage& stage) {
    assert(_parsingKernInfo);
    auto loopType = getLoopType(params);

    if (!_parsingKernInfo->currentLoop && !isLegalTopLoopLevel(loopType)) {
        return tl::make_unexpected(Error{
            .ec = std::error_code(), .desc = "[@kernel] requires at least one [@outer] for-loop"});
    }

    auto* currentLoop = _parsingKernInfo->currentLoop;
    auto& children = currentLoop ? currentLoop->children : _parsingKernInfo->children;
    AttributedLoopTypes parentType{LoopType::Regular};
    if (currentLoop) {
        parentType = currentLoop->type;
    }

    if (!isLegalLoopLevel(loopType, parentType)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(),
                  .desc = "Cannot have [@inner] loop outside of an [@outer] loop"});
    }

    return makeOklLoopInfo(stmt, attr, loopType, *_parsingKernInfo, stage)
        .and_then([&children, this](auto&& loopInfo) -> tl::expected<void, Error> {
            children.emplace_back(loopInfo);

            auto& child = children.back();
            child.parent = _parsingKernInfo->currentLoop;
            _parsingKernInfo->currentLoop = &child;
            _parsingKernInfo->loopMap.emplace(&child.stmt, &child);
            return {};
        });
}

tl::expected<void, Error> OklSemaCtx::stopParsingAttributedForLoop(const clang::Attr& attr,
                                                                   const clang::ForStmt& stmt,
                                                                   const std::any* params) {
    assert(_parsingKernInfo);

    auto loopType = getLoopType(params);
    if (loopType.front() == LoopType::Regular) {
        return {};
    }

    auto loopInfo = getLoopInfo(stmt);
    if (!loopInfo) {
        return {};
    }

    _parsingKernInfo->currentLoop = loopInfo->parent;

    return {};
}

ProgramMetaData& OklSemaCtx::getProgramMetaData() {
    return _programMetaData;
}
const ProgramMetaData& OklSemaCtx::getProgramMetaData() const {
    return _programMetaData;
}
}  // namespace oklt
