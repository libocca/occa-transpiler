#pragma once
#include <string>
#include "oklt/core/target_backends.h"

namespace clang {
class CompilerInstance;
}

namespace oklt {

class TranspilerSession;
constexpr const char INTRINSIC_INCLUDE_FILENAME[] = "okl_intrinsic.h";

void addInstrinsicStub(TranspilerSession &session,
                       clang::CompilerInstance &compiler);


std::vector<std::string> embedInstrinsic(std::string &input,
                                         TargetBackend backend);

}  // namespace oklt
