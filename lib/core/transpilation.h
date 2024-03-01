#pragma once

#include <oklt/core/error.h>
#include "core/transpiler_session/session_stage.h"

#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Refactoring/AtomicChange.h>
#include <deque>
#include <tl/expected.hpp>

namespace clang {
class SourceManager;
}

namespace oklt {
struct EncodedReplacement {
    std::string name;
    clang::tooling::Replacement replacemnt;
    EncodedReplacement() = default;
    EncodedReplacement(std::string_view name, clang::tooling::Replacement replacemnt_)
        : name(name),
          replacemnt(std::move(replacemnt_)) {}
};

struct Transpilation {
    std::string name;
    std::vector<EncodedReplacement> replacemnts;
    Transpilation() = default;
    Transpilation(std::string_view name, std::size_t size)
        : name(name) {
        replacemnts.reserve(size);
    }
};

inline bool isEmpty(const Transpilation& t) {
    return t.replacemnts.empty();
}

struct TranspilationBuilder {
    TranspilationBuilder(const clang::SourceManager& sm, std::string_view name, std::size_t size);
    TranspilationBuilder& addReplacement(std::string_view, clang::tooling::Replacement replacemnt);
    TranspilationBuilder& addReplacement(std::string_view,
                                         const clang::SourceRange&,
                                         std::string_view text);
    TranspilationBuilder& addReplacement(std::string_view,
                                         clang::SourceLocation,
                                         std::string_view text);
    TranspilationBuilder& addReplacement(std::string_view,
                                         clang::SourceLocation,
                                         clang::SourceLocation,
                                         std::string_view text);
    TranspilationBuilder& addInclude(std::string_view);

    tl::expected<Transpilation, Error> build();

   private:
    const clang::SourceManager& _sm;
    Transpilation _trasnpilation;
};

using Transpilations = std::deque<Transpilation>;

tl::expected<std::string, std::error_code> applyTranspilations(const Transpilations&,
                                                               const clang::SourceManager&);
bool applyTranspilations(const Transpilations&, clang::Rewriter&);
bool applyTranspilations(const Transpilations&, SessionStage&);
}  // namespace oklt
