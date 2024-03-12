#pragma once

#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace oklt::serial_subset {
struct ExLoopInfo {
    std::list<std::reference_wrapper<const clang::Decl>> shared = {};
    std::list<std::reference_wrapper<const clang::VarDecl>> exclusive = {};
};

struct BackendCtx {
    std::unordered_map<OklLoopInfo*, ExLoopInfo> loopMap = {};

    ExLoopInfo& getLoopInfo(OklLoopInfo* loopInfo);
};

inline BackendCtx& getBackendCtxFromStage(SessionStage& s) {
    return s.tryEmplaceUserCtx<serial_subset::BackendCtx>();
};

inline ExLoopInfo& BackendCtx::getLoopInfo(OklLoopInfo* loopInfo) {
    return loopMap.try_emplace(loopInfo).first->second;
}

}  // namespace oklt::serial_subset
