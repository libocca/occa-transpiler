#include "okl_attr_traverser.h"
#include <llvm/Support/FormatVariadic.h>

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
  Preprocessor& pp;
};

OklAttributePrarserFsm makeOklAttrParserFsm(Preprocessor& pp, const std::vector<Token>& tokens) {
  return {.token_cursor = 0,
          .state        = OklAttributeParserState::SearchingAttrStart,
          .attr         = {},
          .pp           = pp};
}

void resetFsmAttrState(OklAttributePrarserFsm& fsm) {
  fsm.attr  = {};
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
        llvm::outs() << "malformed okl attr params: " << token.getName() << '\n'
                     << getTokenName(token.getKind()) << " "
                     << token.getLocation().printToString(fsm.pp.getSourceManager()) << '\n';
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

        fsm.state = OklAttributeParserState::ParseAttrParamList;
      } else {
        return FsmStepStatus::OklAttrParsed;
      }
      break;
    case OklAttributeParserState::ParseAttrParamList:
      if (token.is(tok::r_paren)) {
        fsm.attr.tok_indecies.push_back(fsm.token_cursor);
        fsm.attr.params += fsm.pp.getSpelling(token);

        return FsmStepStatus::OklAttrParsed;
      }
      if (token.isOneOf(tok::at, tok::equal, tok::identifier, tok::comma, tok::string_literal,
                        tok::numeric_constant, tok::kw_false, tok::kw_true)) {
        fsm.attr.tok_indecies.push_back(fsm.token_cursor);
        fsm.attr.params += [&](const auto& token) {
          auto token_str = fsm.pp.getSpelling(token);
          return token.getKind() != tok::string_literal
                     ? token_str
                     : std::string(llvm::formatv("\"{0}\"", token_str));
        }(token);
        return FsmStepStatus::TokenProcessed;
      }

      llvm::outs() << "malformed token in attribute param list: " << token.getName() << '\n'
                   << getTokenName(token.getKind()) << " "
                   << token.getLocation().printToString(fsm.pp.getSourceManager()) << '\n';
      return FsmStepStatus::Error;
    default:
      llvm::outs() << "malformed fsm condition in attribute param list\n"
                   << getTokenName(token.getKind()) << " "
                   << token.getLocation().printToString(fsm.pp.getSourceManager()) << '\n';
      return FsmStepStatus::Error;
  }

  return FsmStepStatus::TokenProcessed;
}

int parseAndVisitOklAttrFromTokens(const std::vector<Token>& tokens,
                                   Preprocessor& pp,
                                   OklAttrVisitor& visitor) {
  assert(!tokens.empty());
  if (tokens.empty()) {
    llvm::outs() << "no input tokens\n";
    return -2;
  }

  // set intial FSM state with clear attr data
  auto fsm = makeOklAttrParserFsm(pp, tokens);

  // feed fsm all tokens
  // early termination is possible on malformed OKL attribure syntax
  while (true) {
    if (fsm.token_cursor >= tokens.size()) {
      return 0;
    }
    // process one by one token
    const auto& processing_token = tokens[fsm.token_cursor];
    auto status                  = processTokenByFsm(fsm, processing_token);

    if (status == FsmStepStatus::Error) {
      llvm::outs() << "error during parsing okl attr\n"
                   << tokens[fsm.token_cursor].getLocation().printToString(pp.getSourceManager());
      return -1;
    }

    if (status == FsmStepStatus::TokenProcessed) {
      ++fsm.token_cursor;
      continue;
    }

    // OKL attribute is parsed and is ready to precess by user callback
    if (status == FsmStepStatus::OklAttrParsed) {
      auto cont = visitor(fsm.attr, tokens, pp);
      if (!cont) {
        return 1;
      }

      // reset FSM to parse next OKL attrbute
      resetFsmAttrState(fsm);

      continue;
    }
  }

  return 0;
}
}  // namespace

namespace oklt {
int visitOklAttributes(const std::vector<Token>& tokens,
                       clang::Preprocessor& pp,
                       OklAttrVisitor visitor) {
  return parseAndVisitOklAttrFromTokens(tokens, pp, visitor);
}

}  // namespace oklt
