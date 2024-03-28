#include "pipeline/stages/normalizer/impl/okl_attr_traverser.h"
#include "pipeline/stages/normalizer/error_codes.h"

#include <llvm/Support/FormatVariadic.h>
#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

enum class OklAttributeParserState {
    SearchingAttrStart,
    ParseAttrName,
    SearchingAttrParamList,
    ParseAttrParamList
};

enum class FsmStepStatus { Error = -1, TokenProcessed = 0, OklAttrParsed = 1 };

struct OklAttributePrarserFsm {
    size_t token_cursor{0};
    OklAttributeParserState state{OklAttributeParserState::SearchingAttrStart};
    OklAttribute attr;
    const std::vector<Token>* tokens{nullptr};
    Preprocessor& pp;
    uint32_t parenDepth{0};
};

OklAttributePrarserFsm makeOklAttrParserFsm(Preprocessor& pp, const std::vector<Token>& tokens) {
    return {.token_cursor = 0,
            .state = OklAttributeParserState::SearchingAttrStart,
            .attr = {},
            .tokens = &tokens,
            .pp = pp};
}

bool areAllTokensProcessed(const OklAttributePrarserFsm& fsm) {
    return fsm.token_cursor >= fsm.tokens->size();
}

const Token& getCurrentToken(const OklAttributePrarserFsm& fsm) {
    return fsm.tokens->at(fsm.token_cursor);
}

void incCurrsorToken(OklAttributePrarserFsm& fsm) {
    ++fsm.token_cursor;
}

void resetFsmAttrState(OklAttributePrarserFsm& fsm) {
    fsm.attr = {};
    fsm.state = OklAttributeParserState::SearchingAttrStart;
}

bool isOklAttrMarker(const Token& token) {
    return token.is(tok::at);
}

FsmStepStatus processTokenByFsm(OklAttributePrarserFsm& fsm, const Token& token) {
    switch (fsm.state) {
        case OklAttributeParserState::SearchingAttrStart:
            if (!isOklAttrMarker(token)) {
                break;
            }
            fsm.attr.tok_indecies.push_back(fsm.token_cursor);
            fsm.state = OklAttributeParserState::ParseAttrName;
            break;
        case OklAttributeParserState::ParseAttrName:
            if (token.isNot(tok::identifier)) {
                SPDLOG_ERROR("malformed okl attr params: {} {} {}",
                             token.getName(),
                             getTokenName(token.getKind()),
                             token.getLocation().printToString(fsm.pp.getSourceManager()));
                return FsmStepStatus::Error;
            }

            fsm.attr.name = fsm.pp.getSpelling(token);
            fsm.attr.tok_indecies.push_back(fsm.token_cursor);

            fsm.state = OklAttributeParserState::SearchingAttrParamList;
            break;
        case OklAttributeParserState::SearchingAttrParamList:
            if (token.is(tok::l_paren)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.attr.params += fsm.pp.getSpelling(token);
                fsm.parenDepth += 1u;

                fsm.state = OklAttributeParserState::ParseAttrParamList;
            } else {
                return FsmStepStatus::OklAttrParsed;
            }
            break;
        case OklAttributeParserState::ParseAttrParamList:
            // close current parentness
            if (token.is(tok::r_paren)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.attr.params += fsm.pp.getSpelling(token);
                fsm.parenDepth -= 1u;

                // all parentness are closed norify that OKL attribute is parsed
                if (!fsm.parenDepth) {
                    return FsmStepStatus::OklAttrParsed;
                }

                // still opened parentness so continue parsing
                break;
            }

            // open nested parentness
            if (token.is(tok::l_paren)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.attr.params += fsm.pp.getSpelling(token);
                fsm.parenDepth += 1u;
                break;
            }

            if (token.isOneOf(tok::at,
                              tok::equal,
                              tok::identifier,
                              tok::comma,
                              tok::string_literal,
                              tok::numeric_constant,
                              tok::kw_false,
                              tok::kw_true,
                              tok::slash,
                              tok::star,
                              tok::plus,
                              tok::minus)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.attr.params += [&](const auto& token) {
                    auto token_str = fsm.pp.getSpelling(token);
                    return token.getKind() != tok::string_literal
                               ? token_str
                               : std::string(llvm::formatv("\"{0}\"", token_str));
                }(token);
                return FsmStepStatus::TokenProcessed;
            }

            SPDLOG_ERROR("malformed token in attribute param list: {} {} {}",
                         token.getName(),
                         getTokenName(token.getKind()),
                         token.getLocation().printToString(fsm.pp.getSourceManager()));
            return FsmStepStatus::Error;
        default:
            SPDLOG_ERROR("malformed token in attribute param list: {} {}",
                         getTokenName(token.getKind()),
                         token.getLocation().printToString(fsm.pp.getSourceManager()));
            return FsmStepStatus::Error;
    }

    return FsmStepStatus::TokenProcessed;
}

tl::expected<void, Error> parseAndVisitOklAttrFromTokens(const std::vector<Token>& tokens,
                                                         Preprocessor& pp,
                                                         OklAttrVisitor& visitor) {
    if (tokens.empty()) {
        SPDLOG_CRITICAL("no input tokens");
        return tl::make_unexpected(
            makeError(OkltNormalizerErrorCode::NO_TOKENS_FROM_SOURCE, "no tokens in source"));
    }

    // set intial FSM state with clear attr data
    auto fsm = makeOklAttrParserFsm(pp, tokens);

    // feed fsm all tokens
    // early termination is possible on malformed OKL attribure syntax
    while (true) {
        if (areAllTokensProcessed(fsm)) {
            return {};
        }
        // process one by one token
        const auto& processing_token = getCurrentToken(fsm);
        auto status = processTokenByFsm(fsm, processing_token);

        if (status == FsmStepStatus::Error) {
            SPDLOG_CRITICAL(
                "error during parsing okl attr: {}",
                tokens[fsm.token_cursor].getLocation().printToString(pp.getSourceManager()));
            return tl::make_unexpected(Error{
                OkltNormalizerErrorCode::OKL_ATTR_PARSIN_ERR,
                "error on token at: " +
                    tokens[fsm.token_cursor].getLocation().printToString(pp.getSourceManager())});
        }

        if (status == FsmStepStatus::TokenProcessed) {
            incCurrsorToken(fsm);
            continue;
        }

        // OKL attribute is parsed and is ready to precess by user callback
        if (status == FsmStepStatus::OklAttrParsed) {
            auto cont = visitor(fsm.attr, tokens, pp);
            if (!cont) {
                return {};
            }

            // reset FSM to parse next OKL attrbute
            resetFsmAttrState(fsm);

            continue;
        }
    }

    return {};
}
}  // namespace

namespace oklt {
tl::expected<void, Error> visitOklAttributes(const std::vector<Token>& tokens,
                                             clang::Preprocessor& pp,
                                             OklAttrVisitor visitor) {
    return parseAndVisitOklAttrFromTokens(tokens, pp, visitor);
}

}  // namespace oklt
