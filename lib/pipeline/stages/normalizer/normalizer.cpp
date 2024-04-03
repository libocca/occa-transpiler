#include "impl/expand_macro_stage.h"
#include "impl/gnu_to_std_cpp_stage.h"
#include "impl/okl_macro_stage.h"
#include "impl/okl_to_gnu_stage.h"

#include "pipeline/stages/normalizer/normalizer.h"

#include <llvm/Support/raw_os_ostream.h>
#include <oklt/core/error.h>
#include <spdlog/spdlog.h>

using namespace clang;

namespace {
using namespace oklt;

OklMacroStageInput toOklMacroInput(SharedTranspilerSession session) {
    return {
        .cppSrc = std::move(session->input.sourceCode),
        .session = session,
    };
}

ExpandMacroStageInput toExpandMacroInput(OklMacroStageOutput output) {
    return {
        .cppSrc = std::move(output.cppSrc),
        .cppIncs = std::move(output.cppIncs),
        .session = output.session,
    };
}

OklToGnuStageInput toOkltoGnuInput(ExpandMacroStageOutput& output) {
    return {
        .oklCppSrc = std::move(output.cppSrc),
        .oklCppIncs = std::move(output.cppIncs),
        .session = output.session,
    };
}

GnuToStdCppStageInput toStdCppStageInput(OklToGnuStageOutput& output) {
    return {.gnuCppSrc = std::move(output.gnuCppSrc),
            .gnuCppIncs = std::move(output.gnuCppIncs),
            .gnuMarkers = std::move(output.gnuMarkers),
            .session = output.session};
}

TranspilerSessionResult toSessionResult(GnuToStdCppStageOutput output) {
    // copy to output as final result of normalization stage
    output.session->output.normalized.sourceCode = output.stdCppSrc;

    // pass output as the input for this next stage
    output.session->input.sourceCode = std::move(output.stdCppSrc);
    output.session->normalizedHeaders = std::move(output.stdCppIncs);
    return output.session;
}

OklMacroResult runOklMacroAttrConverter(SharedTranspilerSession session) {
    SPDLOG_INFO("Convert OKL macro attributes");
    return convertOklMacroAttribute(toOklMacroInput(session));
}

ExpandMacroResult runMacroExpander(OklMacroStageOutput output) {
    SPDLOG_INFO("Expand macro");
    return expandMacro(toExpandMacroInput(output));
}

OklToGnuResult runOklToGnuConverter(ExpandMacroStageOutput output) {
    SPDLOG_INFO("Convert OKL attributes to GNU attributes");
    return convertOklToGnuAttribute(toOkltoGnuInput(output));
}

GnuToStdCppResult runGnuToStdConverter(OklToGnuStageOutput output) {
    SPDLOG_INFO("Convert GNU attributes to CPP attributes");
    return convertGnuToStdCppAttribute(toStdCppStageInput(output));
}
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
//
TranspilerSessionResult applyGnuAttrBasedNormalization(SharedTranspilerSession session) {
    return runOklMacroAttrConverter(session)
        .and_then(runMacroExpander)
        .and_then(runOklToGnuConverter)
        .and_then(runGnuToStdConverter)
        .and_then(toSessionResult);
}

}  // namespace
namespace oklt {

TranspilerSessionResult runNormalizerStage(SharedTranspilerSession session) {
    SPDLOG_INFO("Start normalization stage");
    return applyGnuAttrBasedNormalization(session);
}
}  // namespace oklt
