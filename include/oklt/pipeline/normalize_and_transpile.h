#pragma once
#include <oklt/pipeline/stages/transpiler/transpiler.h>

namespace oklt {


tl::expected<TranspilerResult,std::vector<Error>> normalize_and_transpile(TranspilerInput input);

}
