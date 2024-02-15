#include <oklt/attributes/frontend/parsers/kernel.h>

namespace oklt {
bool parseKernelAttribute(const clang::Attr* a, SessionStage&) {
    llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

}  // namespace oklt