#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class SessionStage;

class AttrStmtHandler {
 public:
  using ParamsParserType = std::function<bool(const clang::Attr*, SessionStage&)>;
  using HandleType = std::function<bool(const clang::Attr*, const clang::Stmt*, SessionStage&)>;

  AttrStmtHandler(ParamsParserType pp, HandleType h);
  ~AttrStmtHandler() = default;

  bool handle(const clang::Attr* attr, const clang::Stmt*, SessionStage& session);

 protected:
  bool parseParams(const clang::Attr*, SessionStage& session);

 private:
  ParamsParserType _paramsParser;
  HandleType _handler;
};
}  // namespace oklt
