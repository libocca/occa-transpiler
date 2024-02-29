#include <oklt/core/error.h>
#include <tl/expected.hpp>
#include "core/transpilation.h"

#include <any>

namespace oklt {

using HandleResult = tl::expected<Transpilation, Error>;
using ParseResult = tl::expected<std::any, Error>;

}  // namespace oklt
