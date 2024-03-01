#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"

namespace {
using namespace oklt;
using namespace clang;
using namespace clang::tooling;

tl::expected<Replacements, std::error_code> toReplacements(const Transpilations& transpilations) {
    Replacements replacemnts;
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "applying transpilations\n";
#endif
    for (const auto& t : transpilations) {
#ifdef TRANSPILER_DEBUG_LOG
        llvm::outs() << "applying replacemnt: " << t.name << " - ";
#endif
        for (const auto& r : t.replacemnts) {
#ifdef TRANSPILER_DEBUG_LOG
            llvm::outs() << "applying replacemnt: " << r.name << " - ";
#endif
            auto error = replacemnts.add(r.replacemnt);
            if (error) {
                llvm::errs() << toString(std::move(error));

                Replacements second;
                error = second.add(r.replacemnt);
                if (error) {
                    llvm::errs() << toString(std::move(error));
                    return tl::make_unexpected(std::error_code());
                }

                auto merged = replacemnts.merge(second);
                replacemnts = std::move(merged);
            }
#ifdef TRANSPILER_DEBUG_LOG
            llvm::outs() << "ok.\n";
#endif
        }
    }

    return replacemnts;
}
}  // namespace

namespace oklt {

TranspilationBuilder::TranspilationBuilder(const clang::SourceManager& sm,
                                           std::string_view name,
                                           std::size_t size)
    : _sm(sm),
      _trasnpilation(name, size) {}

TranspilationBuilder& TranspilationBuilder::addReplacement(std::string_view name,
                                                           const clang::SourceRange& sourceRange,
                                                           std::string_view text) {
    _trasnpilation.replacemnts.emplace_back(
        name, clang::tooling::Replacement(_sm, CharSourceRange(sourceRange, true), text));

    return *this;
}

TranspilationBuilder& TranspilationBuilder::addReplacement(std::string_view name,
                                                           clang::SourceLocation startLoc,
                                                           std::string_view text) {
    _trasnpilation.replacemnts.emplace_back(
        name,
        clang::tooling::Replacement(
            _sm, CharSourceRange(SourceRange(startLoc, startLoc), false), text));

    return *this;
}

TranspilationBuilder& TranspilationBuilder::addReplacement(std::string_view name,
                                                           clang::SourceLocation startLoc,
                                                           clang::SourceLocation endLoc,
                                                           std::string_view text) {
    _trasnpilation.replacemnts.emplace_back(
        name,
        clang::tooling::Replacement(
            _sm, CharSourceRange(SourceRange(startLoc, endLoc), true), text));

    return *this;
}

TranspilationBuilder& TranspilationBuilder::addInclude(std::string_view include) {
    auto mainFileId = _sm.getMainFileID();
    auto loc = _sm.getLocForStartOfFile(mainFileId);

    _trasnpilation.replacemnts.emplace_back(
        OKL_INCLUDES,
        clang::tooling::Replacement(_sm,
                                    CharSourceRange(SourceRange(loc, loc), false),
                                    "#include " + std::string(include) + "\n"));

    return *this;
}

tl::expected<Transpilation, Error> TranspilationBuilder::build() {
    return std::move(_trasnpilation);
}

tl::expected<std::string, std::error_code> applyTranspilations(const Transpilations& transpilations,
                                                               const SourceManager& sm) {
    auto result = toReplacements(transpilations);
    if (!result) {
        return tl::make_unexpected(result.error());
    }
    auto source = sm.getBufferData(sm.getMainFileID());

    auto transpiledResult = applyAllReplacements(source, result.value());
    if (!transpiledResult) {
        // TODO convert to std::error
        return tl::make_unexpected(llvm::errorToErrorCode(transpiledResult.takeError()));
    }

    return transpiledResult.get();
}

bool applyTranspilations(const Transpilations& transpilations,
                         Rewriter& rewriter) {
    auto result = toReplacements(transpilations);
    if (!result) {
        return false;
    }

    return applyAllReplacements(result.value(), rewriter);
}

bool applyTranspilations(const Transpilations& transpilations,
                         SessionStage& stage) {
    auto& rewriter = stage.getRewriter();
    auto& sourceManager = stage.getCompiler().getSourceManager();
    auto mainFileId = sourceManager.getMainFileID();

    Replacements replacemnts;
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "applying transpilations\n";
#endif
    for (const auto& t : transpilations) {
#ifdef TRANSPILER_DEBUG_LOG
        llvm::outs() << "applying replacemnt: " << t.name << " - ";
#endif
        for (const auto& r : t.replacemnts) {
#ifdef TRANSPILER_DEBUG_LOG
            llvm::outs() << "applying replacemnt: " << r.name << " - ";
#endif
            SourceLocation beg = sourceManager.getComposedLoc(mainFileId, r.replacemnt.getOffset());
            SourceLocation end = sourceManager.getComposedLoc(mainFileId, r.replacemnt.getOffset() + r.replacemnt.getLength());
            rewriter.ReplaceText(SourceRange(beg, end), r.replacemnt.getReplacementText());
#ifdef TRANSPILER_DEBUG_LOG
            llvm::outs() << "ok.\n";
#endif
        }
    }
    return true;
}


}  // namespace oklt
