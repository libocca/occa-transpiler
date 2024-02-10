#include <oklt/core/ast_traversal/transpile_types/function_info.h>

namespace oklt {

FunctionInfo::FunctionInfo(clang::FunctionDecl *funDecl)
    : astNode(funDecl)
    , attrs()
    , returnType()
    , name()
    , parameters()
    , outForStmts()
    ,_is_valid(false)
{}

void FunctionInfo::makeValid(const std::string &functionName) {
  if(!functionName.empty()) {
    name = functionName;
    _is_valid = true;
  }
}

bool FunctionInfo::isValid() const {
  return _is_valid;
}

//TODO: add implementation
std::vector<ParsedKernelInfo> FunctionInfo::makeParsedKernelInfo() const {
  if(!isValid()) {
    //TODO: internal error
    return {};
  }
  return {};
}

//TODO: add implementation
std::string FunctionInfo::getFunctionSignature() const {
  if(!isValid()) {
    //TODO: internal error
    return {};
  }
  return {};
}

}
