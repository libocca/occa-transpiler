#include <oklt/util/string_utils.h>

#include "attributes/attribute_names.h"
#include "core/transpilation.h"
#include "core/transpilation_decoders.h"
#include "core/transpilation_encoded_names.h"

namespace oklt {
tl::expected<std::string, Error> decodeKernelModifier(const Transpilation& t) {
    if (t.name != KERNEL_ATTR_NAME) {
        return tl::make_unexpected(
            Error{{}, util::fmt("transpilatiion: {} is not for kernel", t.name).value()});
    }

    for (const auto& r : t.replacemnts) {
        if (r.name != OKL_TRANSPILED_ATTR) {
            continue;
        }
        return r.replacemnt.getReplacementText().data();
    }
    return tl::make_unexpected(
        Error{{}, util::fmt("failed to decode kernel modifier {}", t.name).value()});
}

tl::expected<std::string, Error> decodeParamModifier(const Transpilation& t) {
    for (const auto& r : t.replacemnts) {
        if (r.name != OKL_TRANSPILED_ARG) {
            continue;
        }
        return r.replacemnt.getReplacementText().data();
    }
    return tl::make_unexpected(
        Error{{}, util::fmt("failed to decode param modifier {}", t.name).value()});
}
}  // namespace oklt
