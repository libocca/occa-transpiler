#include "parse_loop_attribute_params.h"
#include <oklt/util/string_utils.h>
#include "attributes/frontend/params/inner_outer.h"
#include "attributes/utils/parse.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
ParseResult parseLoopAttrParams(const clang::Attr& a, SessionStage& s, LoopType loopType) {
    auto fail = [&s](const std::string& err) {
        s.pushError(std::error_code(), err);
        return false;
    };

    if (loopType == LoopType::Regular) {
        return fail("Can't parse loop attributes of loop without attributes");
    }

    auto loopTypeStr = loopType == LoopType::Inner ? "@inner" : "@outer";

    auto tileParamsStr = parseOKLAttributeParamsStr(a);
    if (!tileParamsStr.has_value()) {
        s.pushError(tileParamsStr.error());
        return false;
    }

    auto nParams = tileParamsStr->size();
    if (nParams > 1) {
        return fail(util::fmt("{} has 0 to 1 parameters", loopTypeStr).value());
    }

    auto dimIdx = 0;
    if (nParams == 1 && tileParamsStr.value()[0] != "") {
        auto dimIdxOpt = util::parseStrTo<int>(tileParamsStr.value()[0]);
        if (!dimIdxOpt) {
            return fail(util::fmt("{} arguments must be 0, 1 or 2", loopTypeStr).value());
        }
        dimIdx = dimIdxOpt.value();
    }

    AttributedLoop loop{loopType, static_cast<Dim>(dimIdx)};

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Parsed " << loopTypeStr << " parameters with dim "
                 << static_cast<int>(loop.dim) << "\n";
#endif

    return loop;
}
}  // namespace oklt
