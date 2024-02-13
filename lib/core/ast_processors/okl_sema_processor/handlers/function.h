namespace clang {
class FunctionDecl;
class ParmVarDecl;
}

namespace oklt {
struct SessionStage;

bool validateOklKernelFunction(const clang::FunctionDecl* fd, SessionStage& stage);
bool transpileOklKernelFunction(const clang::FunctionDecl* decl, SessionStage& stage);

bool validateOklKernelParam(const clang::ParmVarDecl* fd, SessionStage& stage);
bool transpileOklKernelParam(const clang::ParmVarDecl* decl, SessionStage& stage);
}  // namespace oklt
