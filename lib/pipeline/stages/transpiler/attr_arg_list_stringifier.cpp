#include "core/lex/lexer.h"
#include "core/transpiler_session/session_stage.h"

#include "pipeline/core/error_codes.h"
#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"

#include "pipeline/utils/okl_attribute.h"
#include "pipeline/utils/std_okl_attribute_traverser.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/LiteralSupport.h>

#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

bool stringifyAttributeArgList(const OklAttribute& oklAttr,
                               const std::vector<Token>& tokens,
                               Preprocessor& pp,
                               oklt::Rewriter& rewriter,
                               OriginalSourceMapper& mapper) {
    SourceLocation startLoc = tokens[oklAttr.tok_indecies.front()].getLocation();
    SourceLocation endLoc = tokens[oklAttr.tok_indecies.back()].getLocation();

    rewriter.ReplaceText({startLoc, endLoc},
                         fmt::format("[[{}(\"{}\")]]", oklAttr.name, oklAttr.params));

    SPDLOG_DEBUG("stringify attr arg list: {} at loc: {}",
                 oklAttr.name,
                 startLoc.printToString(pp.getSourceManager()));

    return true;
}

class AttrArgListStringifier : public StageAction {
   public:
    AttrArgListStringifier() { _name = ATTR_ARG_LIST_STRINGIFIER; }

    bool BeginSourceFileAction(clang::CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();
        auto tokens = fetchTokens(pp);

        if (tokens.empty()) {
            _stage->pushError(OkltPipelineErrorCode::EMPTY_SOURCE_STRING, "no tokens in source?");
            return false;
        }

        auto& rewriter = _stage->getRewriter();
        auto result = visitStdOklAttributes(
            tokens,
            pp,
            [this, &rewriter](
                const OklAttribute& attr, const std::vector<Token>& tokens, Preprocessor& pp) {
                return stringifyAttributeArgList(
                    attr, tokens, pp, rewriter, _stage->getSession().getOriginalSourceMapper());
            });
        if (!result) {
            _stage->pushError(result.error().ec, result.error().desc);
            return false;
        }

        return true;
    }

   protected:
    RewriterProxyType getRewriterType() const override { return RewriterProxyType::WithDeltaTree; }
};

StagePluginRegistry::Add<AttrArgListStringifier> attrArgListStringifier(ATTR_ARG_LIST_STRINGIFIER,
                                                                        "");
}  // namespace
