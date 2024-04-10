#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"

#include "core/sema/okl_sema_ctx.h"
#include "core/sema/okl_sema_info.h"
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

struct LoopAxisTypes {
    LoopTypes types;
    Axises axis;
};

LoopAxisTypes getLoopAxisType(const std::any* param) {
    if (!param) {
        return {{LoopType::Regular}, {Axis::Auto}};
    }

    LoopAxisTypes res{};
    if (param->type() == typeid(TileParams)) {
        auto tile = std::any_cast<TileParams>(*param);
        res.types = {tile.firstLoop.type, tile.secondLoop.type};
        res.axis = {tile.firstLoop.axis, tile.secondLoop.axis};
    } else if (param->type() == typeid(AttributedLoop)) {
        auto loop = std::any_cast<AttributedLoop>(*param);
        res.types = {loop.type};
        res.axis = {loop.axis};
    }

    return res;
}

tl::expected<OklLoopInfo, Error> makeOklLoopInfo(const clang::ForStmt& stmt,
                                                 const clang::Attr* attr,
                                                 LoopAxisTypes loopTypeAxis,
                                                 OklSemaCtx::ParsedKernelInfo& kernelInfo,
                                                 SessionStage& stage) {
    auto parsedLoopInfo = parseForStmt(attr, stmt, stage);
    if (!parsedLoopInfo) {
        return parsedLoopInfo;
    }
    parsedLoopInfo->type = loopTypeAxis.types;
    parsedLoopInfo->axis = loopTypeAxis.axis;
    return parsedLoopInfo;
}

// check if loop types inside one loop are legal. firstType/lastType - first and alst non regular
// loop type
bool isLegalLoopLevel(LoopTypes loopTypes, LoopType& firstType, LoopType& lastType) {
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

bool isLegalLoopLevel(LoopTypes childTypes, LoopTypes parentTypes = {LoopType::Regular}) {
    LoopType firstParentType = LoopType::Regular, lastParentType = LoopType::Regular;
    LoopType firstChildType = LoopType::Regular, lastChildType = LoopType::Regular;
    if (!isLegalLoopLevel(parentTypes, firstParentType, lastParentType)) {
        return false;
    }
    if (!isLegalLoopLevel(childTypes, firstChildType, lastChildType)) {
        return false;
    }

    if (lastParentType == firstChildType) {
        return true;
    }

    if (firstChildType == LoopType::Regular || lastParentType == LoopType::Regular) {
        return true;
    }

    if (lastParentType == LoopType::Outer && firstChildType == LoopType::Inner) {
        return true;
    }

    return false;
}

bool isLegalTopLoopLevel(LoopTypes loopType) {
    return loopType.front() == LoopType::Outer;
}

bool isRegular(const LoopAxisTypes& axisTypes) {
    for (auto& type : axisTypes.types) {
        if (type != LoopType::Regular) {
            return false;
        }
    }
    return true;
}

bool isTopLevelAttributed(const LoopAxisTypes& axisTypes,
                          const OklSemaCtx::ParsedKernelInfo& parsedKernelInfo) {
    if (isRegular(axisTypes)) {
        return false;
    }
    auto* currLoop = parsedKernelInfo.currentLoop;
    while (currLoop) {
        if (!currLoop->isRegular()) {
            return false;
        }
        currLoop = currLoop->parent;
    }
    return true;
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

tl::expected<void, Error> OklSemaCtx::startParsingAttributedForLoop(const clang::Attr* attr,
                                                                    const clang::ForStmt& stmt,
                                                                    const std::any* params,
                                                                    SessionStage& stage) {
    if (!_parsingKernInfo) {
        // NOTE: original OKL silently removes attribute
        return tl::make_unexpected(
            Error{std::error_code{}, "Attributed loop outside [@kernel] function"});
    }
    auto loopTypeAxis = getLoopAxisType(params);

    // TODO: currently missing diagnostic on at least one [@outer] loop must be present
    auto reg = isRegular(loopTypeAxis);
    auto* currentLoop = _parsingKernInfo->currentLoop;
    auto isTopLevel = static_cast<bool>(currentLoop);  // for readibility

    auto& children = [&]() -> std::list<OklLoopInfo>& {
        if (isTopLevel) {
            return currentLoop->children;
        }
        return _parsingKernInfo->topLevelLoops;
    }();
    LoopTypes parentType{LoopType::Regular};
    if (currentLoop) {
        parentType = currentLoop->type;
    }

    if (!isLegalLoopLevel(loopTypeAxis.types, parentType)) {
        return tl::make_unexpected(
            Error{.ec = std::error_code(),
                  .desc = "Cannot have [@inner] loop outside of an [@outer] loop"});
    }

    return makeOklLoopInfo(stmt, attr, loopTypeAxis, *_parsingKernInfo, stage)
        .and_then([&](auto&& loopInfo) -> tl::expected<void, Error> {
            children.emplace_back(loopInfo);

            auto& child = children.back();
            if (isTopLevelAttributed(loopTypeAxis, *_parsingKernInfo)) {
                _parsingKernInfo->topLevelOuterLoops.push_back(&child);
            }
            child.parent = _parsingKernInfo->currentLoop;
            _parsingKernInfo->currentLoop = &child;
            if (_parsingKernInfo->loopMap.count(&child.stmt)) {
                return tl::make_unexpected(
                    Error{std::error_code(), "Multiple attributes on one loop"});
            }
            _parsingKernInfo->loopMap.emplace(&child.stmt, &child);
            return {};
        });
}

tl::expected<void, Error> OklSemaCtx::stopParsingAttributedForLoop(const clang::Attr* attr,
                                                                   const clang::ForStmt& stmt,
                                                                   const std::any* params) {
    assert(_parsingKernInfo);

    auto loopInfo = getLoopInfo(stmt);
    if (!loopInfo) {
        return {};
    }
    // Set specific axis here, since children for loopInfo should be complete
    if (loopInfo->has(Axis::Auto)) {
        if (!loopInfo->updateAutoWithSpecificAxis()) {
            return tl::make_unexpected(
                Error{{}, util::fmt("More than {} nested [@inner] loops", N_AXIS).value()});
        }
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
