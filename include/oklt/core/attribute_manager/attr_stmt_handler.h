#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class TranspileSession;

class AttrStmtHandler {
public:
  using ParamsParserType = std::function<bool(const clang::Attr *, TranspileSession &)>;
  using HandleType = std::function<bool(const clang::Attr*, const clang::Stmt*, TranspileSession &)>;

  AttrStmtHandler(ParamsParserType pp, HandleType h);
  ~AttrStmtHandler() = default;

  bool handle(const clang::Attr* attr, const clang::Stmt*, TranspileSession &session);
protected:
  bool parseParams(const clang::Attr*, TranspileSession &session);
private:
  ParamsParserType _paramsParser;
  HandleType _handler;
};
}
