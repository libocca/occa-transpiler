#pragma once

#include <string>
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/loop_meta_data.h"

namespace oklt::cuda_subset {
std::string dimToStr(const Dim& dim);
std::string getIdxVariable(const AttributedLoop& loop);
std::string getCondCompStr(const LoopMetadata& forLoop);
std::string getUnaryStr(const LoopMetadata& forLoop, const std::string& var);
std::string buildCloseScopes(int& openedScopeCounter);
void replaceAttributedLoop(const clang::Attr* a,
                           const clang::ForStmt* f,
                           const std::string& prefixCode,
                           const std::string& suffixCode, 
                           SessionStage& s);

namespace tile {
std::string getTiledVariableName(const LoopMetadata& forLoop);

// Produces something like: int _occa_tiled_i = init +- ((tileSize * inc) * threadIdx.x);
//                      or: int _occa_tiled_i = init +- ((tileSize * inc) * blockIdx.x);
std::string buildIinnerOuterLoopIdxLineFirst(const LoopMetadata& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter);

// Produces something like: int i = _occa_tiled_i +- (inc * threadIdx.x);
//                      or: int i = _occa_tiled_i +- (inc * blockIdx.x);
std::string buildInnerOuterLoopIdxLineSecond(const LoopMetadata& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter);

// Produces something like:
//      for (int _occa_tiled_i = start; _occa_tiled_i < end; _occa_tiled_i += tileSize) {
// or:  for (int _occa_tiled_i = start; _occa_tiled_i > end; _occa_tiled_i -= tileSize) {
std::string buildRegularLoopIdxLineFirst(const LoopMetadata& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter);

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); i+=inc)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); --i)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); i-=inc)
std::string buildRegularLoopIdxLineSecond(const LoopMetadata& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter);
}  // namespace tile

namespace inner_outer {
// Produces something like: int i = start +- (inc * threadIdx.x);
//                      or: int i = start +- (inc * blockIdx.x);
std::string buildInnerOuterLoopIdxLine(const LoopMetadata& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter);
}  // namespace inner_outer
}  // namespace oklt::cuda_subset