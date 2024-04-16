#include "pipeline/core/error_codes.h"
#include "pipeline/utils/okl_attribute.h"
#include "pipeline/utils/okl_attribute_traverser.h"

#include <clang/Lex/Preprocessor.h>
#include <llvm/Support/FormatVariadic.h>

#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

enum class OklAttributeParserState {
    SearchingFirstOpenBracer,
    SearchingSecondOpenBracer,
    ParseAttrName,
    SearchingAttrParamList,
    ParseAttrParamList,
    SearchingFirstCloseBracer,
    SearchingSecondCloseBracer,
};

enum class FsmStepStatus { Error = -1, TokenProcessed = 0, OklAttrParsed = 1 };

struct OklAttributePrarserFsm {
    size_t token_cursor{0};
    OklAttributeParserState state{OklAttributeParserState::SearchingFirstOpenBracer};
    OklAttribute attr;
    const std::vector<Token>* tokens{nullptr};
    Preprocessor& pp;
    uint32_t parenDepth{0};
};

OklAttributePrarserFsm makeOklAttrParserFsm(Preprocessor& pp, const std::vector<Token>& tokens) {
    return {.token_cursor = 0,
            .state = OklAttributeParserState::SearchingFirstOpenBracer,
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
    fsm.state = OklAttributeParserState::SearchingFirstOpenBracer;
}

bool isCxxOpenAttrBracer(const Token& token) {
    return token.is(tok::l_square);
}

bool isCxxCloseAttrBracer(const Token& token) {
    return token.is(tok::r_square);
}

FsmStepStatus processTokenByFsm(OklAttributePrarserFsm& fsm, const Token& token) {
    switch (fsm.state) {
        case OklAttributeParserState::SearchingFirstOpenBracer:
            if (!isCxxOpenAttrBracer(token)) {
                break;
            }
            fsm.attr.tok_indecies.push_back(fsm.token_cursor);
            fsm.state = OklAttributeParserState::SearchingSecondOpenBracer;
            break;
        case OklAttributeParserState::SearchingSecondOpenBracer:
            if (!isCxxOpenAttrBracer(token)) {
                resetFsmAttrState(fsm);
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
            } else if (token.is(tok::r_square)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.state = OklAttributeParserState::SearchingSecondCloseBracer;
            } else {
                SPDLOG_ERROR("malformed okl attr params: {} {} {}",
                             token.getName(),
                             getTokenName(token.getKind()),
                             token.getLocation().printToString(fsm.pp.getSourceManager()));
                return FsmStepStatus::Error;
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
                    fsm.state = OklAttributeParserState::SearchingFirstCloseBracer;
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
                              tok::r_square,
                              tok::l_square,
                              tok::plus,
                              tok::minus)) {
                fsm.attr.tok_indecies.push_back(fsm.token_cursor);
                fsm.attr.params += [&](const auto& token) {
                    auto token_str = fsm.pp.getSpelling(token);
                    return token.getKind() != tok::string_literal
                               ? token_str
                               : std::string(llvm::formatv("\"{0}\"", token_str));
                }(token);
                break;
            }

            SPDLOG_ERROR("malformed token in attribute param list: {} {} {}",
                         token.getName(),
                         getTokenName(token.getKind()),
                         token.getLocation().printToString(fsm.pp.getSourceManager()));
            return FsmStepStatus::Error;
        case OklAttributeParserState::SearchingFirstCloseBracer:
            if (!isCxxCloseAttrBracer(token)) {
                resetFsmAttrState(fsm);
                break;
            }
            fsm.attr.tok_indecies.push_back(fsm.token_cursor);
            fsm.state = OklAttributeParserState::SearchingSecondCloseBracer;
            break;

        case OklAttributeParserState::SearchingSecondCloseBracer:
            if (!isCxxCloseAttrBracer(token)) {
                resetFsmAttrState(fsm);
                break;
            }
            fsm.attr.tok_indecies.push_back(fsm.token_cursor);
            return FsmStepStatus::OklAttrParsed;
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
            makeError(OkltPipelineErrorCode::NO_TOKENS_FROM_SOURCE, "no tokens in source"));
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
                OkltPipelineErrorCode::OKL_ATTR_PARSING_ERR,
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
tl::expected<void, Error> visitStdOklAttributes(const std::vector<Token>& tokens,
                                                clang::Preprocessor& pp,
                                                OklAttrVisitor visitor) {
    return parseAndVisitOklAttrFromTokens(tokens, pp, visitor);
}

}  // namespace oklt
