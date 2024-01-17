#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class TranspileSession;

class AttrDeclHandler {
public:
  using ParamsParserType = std::function<bool(const clang::Attr *, TranspileSession &)>;
  using HandleType = std::function<bool(const clang::Attr*, const clang::Decl*, TranspileSession &)>;

  AttrDeclHandler(ParamsParserType pp, HandleType h);
  ~AttrDeclHandler() = default;

  bool handle(const clang::Attr* attr, const clang::Decl*, TranspileSession &session);
protected:
  bool parseParams(const clang::Attr*, TranspileSession &session);
private:
  ParamsParserType _paramsParser;
  HandleType _handler;
};
}
