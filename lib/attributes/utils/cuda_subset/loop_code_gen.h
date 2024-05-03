#pragma once

#include "attributes/frontend/params/tile.h"
#include "core/handler_manager/result.h"
#include "core/transpiler_session/session_stage.h"

#include <string>

namespace oklt {
struct OklLoopInfo;
}

namespace oklt::cuda_subset {
std::string axisToStr(const Axis& axis);
std::string getIdxVariable(const AttributedLoop& loop);

namespace tile {
std::string getTiledVariableName(const OklLoopInfo& forLoop);

// Produces something like: int _occa_tiled_i = init +- ((tileSize * inc) * threadIdx.x);
//                      or: int _occa_tiled_i = init +- ((tileSize * inc) * blockIdx.x);
std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter);

// Produces something like: int i = _occa_tiled_i +- (inc * threadIdx.x);
//                      or: int i = _occa_tiled_i +- (inc * blockIdx.x);
std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter);

// Produces something like:
//      for (int _occa_tiled_i = start; _occa_tiled_i < end; _occa_tiled_i += tileSize) {
// or:  for (int _occa_tiled_i = start; _occa_tiled_i > end; _occa_tiled_i -= tileSize) {
std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter,
                                         oklt::Rewriter& rewriter);

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); i+=inc)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); --i)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); i-=inc)
std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter,
                                          oklt::Rewriter& rewriter);
}  // namespace tile

namespace inner_outer {
// Produces something like: int i = start +- (inc * threadIdx.x);
//                      or: int i = start +- (inc * blockIdx.x);
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       oklt::Rewriter& rewriter);
}  // namespace inner_outer
}  // namespace oklt::cuda_subset
