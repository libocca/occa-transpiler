#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/TargetInfo.h>
#include <clang/AST/Attr.h>
#include <clang/Basic/CharInfo.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/LiteralSupport.h>
#include <clang/Lex/Preprocessor.h>

#include <llvm/ADT/StringExtras.h>

namespace {
using namespace oklt;
using namespace clang;
using namespace llvm;

const std::string OKL_ATTRIBUTE_PREFIX = "okl_";

struct BraceCounter {
    unsigned short paren = 0;
    unsigned short bracket = 0;
    unsigned short brace = 0;
    explicit operator bool() const { return (paren != 0 || bracket != 0 || brace != 0); }
    void count(const Token& tok) {
        auto kind = tok.getKind();
        switch (kind) {
            case tok::l_paren:
                ++paren;
                break;
            case tok::r_paren:
                --paren;
                break;
            case tok::l_square:
                ++bracket;
                break;
            case tok::r_square:
                --bracket;
                break;
            case tok::l_brace:
                ++brace;
                break;
            case tok::r_brace:
                --brace;
                break;
            default:
                break;
        }
    }
};

unsigned getIntegerWidth(CharLiteralParser& Literal, const TargetInfo& TI) {
    if (Literal.isMultiChar()) {
        return TI.getIntWidth();
    } else if (Literal.isWide()) {
        return TI.getWCharWidth();
    } else if (Literal.isUTF16()) {
        return TI.getChar16Width();
    } else if (Literal.isUTF32()) {
        return TI.getChar32Width();
    } else {
        // char or char8_t
        return TI.getCharWidth();
    }
}

unsigned getIntegerWidth(NumericLiteralParser& Literal, const TargetInfo& TI) {
    if (Literal.isBitInt) {
        return TI.getMaxBitIntWidth();
    } else if (Literal.isLong) {
        return TI.getLongWidth();
    } else if (Literal.isLongLong || Literal.isSizeT) {
        return TI.getLongLongWidth();
    } else {
        return TI.getIntWidth();
    }
}

const llvm::fltSemantics& getFloatFormat(NumericLiteralParser& Literal, const TargetInfo& TI) {
    assert(Literal.isFloatingLiteral() && "Expected floating point literal");
    if (Literal.isHalf) {  // float16_t ?
        return TI.getFloatFormat();  // return TI.getHalfFormat();
    } else if (Literal.isFloat) {  // float, float32_t
        return TI.getFloatFormat();
    } else if (Literal.isFloat16) {  // float16_t
        return TI.getFloatFormat();
        //        return TI.getBFloat16Format();
    } else if (Literal.isFloat128) {  // float128_t
        return TI.getFloat128Format();  // return TI.getLongDoubleFormat();
    } else {
        return TI.getDoubleFormat();  // double, float64_t
    }
}

class AttrParamParser {
   private:
    using TokenStream = SmallVector<Token, 4>;
    Preprocessor& PP;
    SourceManager& SM;

    std::string strVal = {};
    TokenStream Toks = {};
    TokenStream::const_iterator TokIt = {};

    std::optional<OKLAttrParam> parseArg() {
        if (TokIt == Toks.end()) {
            return std::nullopt;
        }

        if (TokIt->is(tok::at)) {
            auto peekTokIt = std::next(TokIt);
            if (peekTokIt != Toks.end() && peekTokIt->is(tok::raw_identifier)) {
                return parseOKLAttr();
            }
        }

        bool isExpr = false;
        tok::TokenKind tokKind = tok::unknown;
        TokenStream argToks;

        BraceCounter cnt;
        while (TokIt != Toks.end()) {
            argToks.push_back(*TokIt);

            cnt.count(*TokIt);

            if (tokKind == tok::unknown) {
                tokKind = TokIt->getKind();
            } else if ((tokKind == tok::minus || tokKind == tok::plus) &&
                       TokIt->getKind() == tok::numeric_constant) {
                tokKind = TokIt->getKind();
            } else if (TokIt->getKind() != tokKind) {
                isExpr = true;
            }

            ++TokIt;

            if (TokIt->isOneOf(tok::comma, tok::r_paren) && !cnt) {
                break;
            }
        };

        if (argToks.empty()) {
            return std::make_optional<OKLAttrParam>(StringRef(), std::any{});
        }

        const auto* rawStart = SM.getCharacterData(argToks.front().getLocation());
        const auto* rawEnd = SM.getCharacterData(argToks.back().getEndLoc());
        auto rawBuf = StringRef(rawStart, rawEnd - rawStart);

        if (!isExpr) {
            auto t = argToks.front();

            const auto* litStart = SM.getCharacterData(t.getLocation());
            const auto* litEnd = SM.getCharacterData(argToks.back().getEndLoc());
            auto litBuf = StringRef(litStart, litEnd - litStart);

            if (t.is(tok::raw_identifier) && !t.getRawIdentifier().empty()) {
                PP.LookUpIdentifierInfo(t);
                tokKind = t.getKind();
            }

            if (argToks.size() > 1 && tokKind == tok::numeric_constant &&
                (t.getKind() == tok::minus || t.getKind() == tok::plus)) {
                t = argToks[1];
                tokKind = t.getKind();

                const auto* start = SM.getCharacterData(t.getLocation());
                litBuf = StringRef(start, litEnd - start);
            }

            switch (tokKind) {
                case tok::numeric_constant: {
                    auto lit = NumericLiteralParser(litBuf,
                                                    t.getLocation(),
                                                    PP.getSourceManager(),
                                                    PP.getLangOpts(),
                                                    PP.getTargetInfo(),
                                                    PP.getDiagnostics());
                    if (lit.hadError) {
                        break;  // ExprError();
                    }

                    if (lit.isFloatingLiteral()) {
                        APFloat val(getFloatFormat(lit, PP.getTargetInfo()));
                        lit.GetFloatValue(val);
                        if (argToks.front().is(tok::minus)) {
                            val = neg(val);
                        }
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }

                    if (lit.isIntegerLiteral()) {
                        APSInt val(getIntegerWidth(lit, PP.getTargetInfo()), lit.isUnsigned);
                        lit.GetIntegerValue(val);
                        if (argToks.front().is(tok::minus)) {
                            val.negate();
                        }
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }

                    // lit.isFixedPointLiteral()

                    break;
                };

                case tok::char_constant:          // 'x'
                case tok::wide_char_constant:     // L'x'
                case tok::utf8_char_constant:     // u8'x'
                case tok::utf16_char_constant:    // u'x'
                case tok::utf32_char_constant: {  // U'x'
                    auto lit = CharLiteralParser(
                        litBuf.begin(), litBuf.end(), t.getLocation(), PP, t.getKind());
                    if (lit.hadError()) {
                        break;  // ExprError();
                    }

                    auto val = APSInt(getIntegerWidth(lit, PP.getTargetInfo()));
                    val = lit.getValue();

                    // Set Signed/Unsigned flag.
                    const auto& TI = PP.getTargetInfo();
                    if (lit.isWide()) {
                        val.setIsUnsigned(!TargetInfo::isTypeSigned(TI.getWCharType()));
                    } else if (lit.isUTF16() || lit.isUTF32()) {
                        val.setIsUnsigned(true);
                    } else if (lit.isUTF8()) {
                        if (PP.getLangOpts().CPlusPlus) {
                            val.setIsUnsigned(PP.getLangOpts().Char8 != 0 ||
                                              !PP.getLangOpts().CharIsSigned);
                        } else {
                            val.setIsUnsigned(true);
                        }
                    } else {
                        val.setIsUnsigned(!PP.getLangOpts().CharIsSigned);
                    }

                    return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                };

                case tok::string_literal:          // "x"
                case tok::wide_string_literal:     // L"x"
                case tok::utf8_string_literal:     // u8"x"
                case tok::utf16_string_literal:    // u"x"
                case tok::utf32_string_literal: {  // U"x"
                    auto lit = StringLiteralParser(argToks, PP);
                    if (lit.hadError) {
                        break;  // ExprError();
                    }

                    auto ptr = reinterpret_cast<const void*>(lit.GetString().data());
                    auto n = lit.GetNumStringChars();
                    if (lit.isOrdinary() || lit.isUTF8() || lit.isPascal()) {
                        auto val = std::string(lit.GetString());
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }
                    if (lit.isWide()) {
                        auto val = std::wstring(reinterpret_cast<const wchar_t*>(ptr), n);
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }
                    if (lit.isUTF16()) {
                        auto val = std::u16string(reinterpret_cast<const char16_t*>(ptr), n);
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }
                    if (lit.isUTF32()) {
                        auto val = std::u32string(reinterpret_cast<const char32_t*>(ptr), n);
                        return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                    }

                    break;
                };

                case tok::kw_true:
                case tok::kw_false: {
                    auto val = APSInt(std::numeric_limits<bool>::digits);
                    val = t.is(tok::kw_true);

                    return std::make_optional<OKLAttrParam>(rawBuf, std::move(val));
                };

                default:
                    break;
            }
        }

        return std::make_optional<OKLAttrParam>(rawBuf, std::any{});
    }

    std::optional<OKLAttrParam> parseOKLAttr() {
        if (TokIt == Toks.end() || !TokIt->is(tok::at)) {
            return std::nullopt;
        }

        const auto* rawStart = SM.getCharacterData(TokIt->getLocation());

        ++TokIt;
        if (TokIt == Toks.end() || !TokIt->is(tok::raw_identifier)) {
            return std::nullopt;
        }

        auto name = OKL_ATTRIBUTE_PREFIX + TokIt->getRawIdentifier().str();
        auto buffer = StringRef(rawStart, SM.getCharacterData(TokIt->getEndLoc()) - rawStart);

        ++TokIt;

        OKLParsedAttr ret = parseOKLAttr(name);
        if (TokIt == Toks.end()) {
            const auto* rawEnd = SM.getCharacterData(Toks.back().getEndLoc());
            buffer = StringRef(rawStart, rawEnd - rawStart);
        } else {
            auto endIt = std::prev(TokIt);
            const auto* rawEnd = SM.getCharacterData(endIt->getEndLoc());
            buffer = StringRef(rawStart, rawEnd - rawStart);
        }

        return std::make_optional<OKLAttrParam>(buffer, std::move(ret));
    }

   public:
    OKLParsedAttr parseOKLAttr(StringRef attrFullName) {
        OKLParsedAttr ret;
        if (!attrFullName.empty()) {
            ret.name = attrFullName;
        }

        if (TokIt == Toks.end() || !TokIt->is(tok::l_paren)) {
            return ret;
        }

        ++TokIt;

        while (TokIt != Toks.end()) {
            if (TokIt->is(tok::r_paren)) {
                ++TokIt;
                break;
            }

            bool isParsed = false;

            // Keyword arg
            if (TokIt->is(tok::raw_identifier)) {
                StringRef name = TokIt->getRawIdentifier();
                auto peekTokIt = std::next(TokIt);
                if (peekTokIt != Toks.end() && peekTokIt->is(tok::equal)) {
                    std::advance(TokIt, 2);  // tok::identifier, tok::equal
                    auto v = parseArg();
                    if (!v.has_value()) {
                        break;
                    }

                    ret.kwargs.insert_or_assign(name.str(), v.value());
                    isParsed = true;
                }
            }

            // Regular arg
            if (!isParsed) {
                auto v = parseArg();
                if (!v.has_value()) {
                    break;
                }

                ret.args.push_back(std::move(v.value()));
            }

            if (TokIt->is(tok::comma)) {
                ++TokIt;
                continue;
            }

            // Something gone wrong?
            if (TokIt->isNot(tok::r_paren)) {
                break;
            }
        }

        return ret;
    }

    AttrParamParser(const clang::Attr& attr, Preprocessor& PP)
        : PP(PP),
          SM(PP.getSourceManager()) {
        assert((isa<AnnotateAttr>(attr) || isa<SuppressAttr>(attr)) &&
               "We only support AnnotateAttr or SuppressAttr");

        // Get payload
        auto attrRange = attr.getRange();
        if (const auto anno = dyn_cast_or_null<AnnotateAttr>(&attr)) {
            strVal = anno->getAnnotation().str();
        } else if (auto sup = dyn_cast_or_null<SuppressAttr>(&attr)) {
            strVal = sup->diagnosticIdentifiers_begin()->str();
        }

        // Check if there is payload
        if (strVal.empty()) {
            TokIt = Toks.begin();
            return;
        }

        // Prepare Buffer and virtual location
        auto tmpTok = Token{};
        tmpTok.startToken();
        PP.CreateString(strVal, tmpTok);
        auto tokLoc = tmpTok.getLocation();

        // Prepare LangOpts to also parse tok::at
        auto langOpts = PP.getLangOpts();
        langOpts.ObjC = 1;

        llvm::StringRef strView(strVal);
        auto lexer = std::make_unique<Lexer>(
            SM.getFileLoc(tokLoc), langOpts, strView.begin(), strView.begin(), strView.end());

        // Lex tokens
        auto tok = Token{};
        do {
            lexer->LexFromRawLexer(tok);
            if (tok.is(tok::eof))
                break;

            Toks.push_back(tok);
        } while (lexer->getBufferLocation() < strView.end());

        TokIt = Toks.begin();
    }
};

}  // namespace

namespace oklt {
using namespace clang;

OKLParsedAttr ParseOKLAttr(const clang::Attr& attr, SessionStage& stage) {
    assert((isa<AnnotateAttr>(attr) || isa<SuppressAttr>(attr)) &&
           "We only support AnnotateAttr or SuppressAttr");

    auto& PP = stage.getCompiler().getPreprocessor();
    auto parser = AttrParamParser(attr, PP);

    return parser.parseOKLAttr(attr.getNormalizedFullName());
}

}  // namespace oklt
