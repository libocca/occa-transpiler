#include "core/utils/attributes.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace oklt {
using namespace clang;

bool removeAttribute(const clang::Attr& attr, SessionStage& stage) {
    auto& rewriter = stage.getRewriter();
    auto range = getAttrFullSourceRange(attr);
    // INFO: sometimes rewrite functions does the job but return false value
    rewriter.RemoveText(range);
    return true;
}

const std::string OKL_GNU_PREFIX = "okl_";
const std::string OKL_CXX_PREFIX = "okl::";

constexpr SourceLocation::IntTy CXX11_ATTR_PREFIX_LEN = std::char_traits<char>::length("[[");
constexpr SourceLocation::IntTy CXX11_ATTR_SUFFIX_LEN = std::char_traits<char>::length("]]");
constexpr SourceLocation::IntTy GNU_ATTR_PREFIX_LEN =
    std::char_traits<char>::length("__attribute__((");
constexpr SourceLocation::IntTy GNU_ATTR_SUFFIX_LEN = std::char_traits<char>::length("))");

SourceRange getAttrFullSourceRange(const Attr& attr) {
    auto arange = attr.getRange();

    if (attr.isCXX11Attribute() || attr.isC2xAttribute()) {
        arange.setBegin(arange.getBegin().getLocWithOffset(-CXX11_ATTR_PREFIX_LEN));
        arange.setEnd(arange.getEnd().getLocWithOffset(CXX11_ATTR_SUFFIX_LEN));
    }

    if (attr.isGNUAttribute()) {
        arange.setBegin(arange.getBegin().getLocWithOffset(-GNU_ATTR_PREFIX_LEN));
        arange.setEnd(arange.getEnd().getLocWithOffset(GNU_ATTR_SUFFIX_LEN));
    }

    return arange;
}

bool isOklAttribute(const clang::Attr& attr) {
    if (!isa<AnnotateAttr, AnnotateTypeAttr, SuppressAttr>(attr)) {
        return false;
    }
    return StringRef(attr.getNormalizedFullName()).starts_with(OKL_GNU_PREFIX) ||
           StringRef(attr.getNormalizedFullName()).starts_with(OKL_CXX_PREFIX);
}

}  // namespace oklt
