#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/util/io_helper.h>

#include <llvm/Support/raw_os_ostream.h>

#include "impl/gnu_to_std_cpp_stage.h"
#include "impl/okl_to_gnu_stage.h"

using namespace clang;

namespace {
using namespace oklt;
// Normalization is done in two steps:
// 1. Replace all OKL specific attributes by GNU C++ attributes comments
//    at the same source location.
//    It's done in following way:
//      - split original source code into tokens;
//      - traverse tokens and parse okl attribute;
//      - Parsed attribute is replaced by GNU C++ attribute at same location
//      GNU C++ attribute is used due to it can be placed at beginning and end of stmt/decl as
//      OKL attribute
//  2. Normalization replace GNU C++ attributes by standard C++ attribute with unified location.
//     After step 1 new source code is legal C++ and could be parsed in clang AST.
//     AST is traversed and each stmt/decl is tested against GNU attribute with OKL prefix.
//     Then GNU C++ attribute is replaced by standard C++ attribute at the beginning of stmt/expr
//  NOTE
//  There is single corner case when @tile, @outer or @inner attributes can be placed inside of for
//  stmt:
//   for (int i=0; i<N; ++i; @outer)
//  It's invalid C++ syntax and on first step the source code is transformed into following:
//   for (int i=0; i<N; ++i)
//  with saving source location of original attribute. After AST is parsed each source range for
//  'for' stmt is tested against stored corner case to restore OKL attribute as C++ one.
tl::expected<NormalizerOutput, NormalizerError>
applyGnuAttrBasedNormalization(NormalizerInput input, TranspilerSession &session) {
  // TODO error handling
  //
#ifdef NORMALIZER_DEBUG_LOG
  llvm::outs() << "stage 0 OKL source:\n\n" << input.oklSource << '\n';
#endif

  auto firstStageOutput =
      convertOklToGnuAttribute({.oklCppSrc = std::move(input.oklSource)}, session);

  if (!firstStageOutput) {
    llvm::outs() << "error " << firstStageOutput.error() << " on first stage of normalizer\n";
    return tl::make_unexpected(NormalizerError{});
  }

#ifdef NORMALIZER_DEBUG_LOG
  llvm::outs() << "stage 1 GNU cpp source:\n\n" << firstStageOutput->gnuCppSrc << '\n';
#endif

  auto secondStageOutput =
      convertGnuToStdCppAttribute({.gnuCppSrc = std::move(firstStageOutput->gnuCppSrc),
                                   .gnuMarkers = std::move(firstStageOutput->gnuMarkers),
                                   .recoveryMarkers = std::move(firstStageOutput->recoveryMarkers)},
                                  session);

  if (!secondStageOutput) {
    llvm::outs() << "error " << secondStageOutput.error() << " on second stage of normalizer\n";
    return tl::make_unexpected(NormalizerError{});
  }

#ifdef NORMALIZER_DEBUG_LOG
  llvm::outs() << "stage 2 STD cpp source:\n\n" << secondStageOutput->stdCppSrc << '\n';
#endif

  return NormalizerOutput{.cppSource = std::move(secondStageOutput->stdCppSrc)};
}

} // namespace
namespace oklt {

tl::expected<NormalizerOutput, NormalizerError> normalize(NormalizerInput input,
                                                          oklt::TranspilerSession &session) {
  return applyGnuAttrBasedNormalization(std::move(input), session);
}
} // namespace oklt
