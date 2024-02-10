#pragma once

#include <clang/AST/Attr.h>
#include <functional>

namespace oklt {

class SessionStage;

class AttrDeclHandler {
 public:
  using ParamsParserType = std::function<bool(const clang::Attr*, SessionStage&)>;
  using HandleType = std::function<bool(const clang::Attr*,
                                        const clang::Decl*,
                                        SessionStage&)>;

  AttrDeclHandler(ParamsParserType pp, HandleType h);
  ~AttrDeclHandler() = default;

  bool handle(const clang::Attr* attr,
              const clang::Decl*,
              SessionStage& stage);

 protected:
  bool parseParams(const clang::Attr*, SessionStage& stage);

 private:
  ParamsParserType _paramsParser;
  HandleType _handler;
};
}  // namespace oklt
