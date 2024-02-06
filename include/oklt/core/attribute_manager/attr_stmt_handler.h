#pragma once

#include <clang/AST/Attr.h>
#include <functional>
#include <oklt/core/attribute_manager/transpile_changes.h>

namespace oklt {

class SessionStage;

class AttrStmtHandler {
 public:
  using ParamsParserType = std::function<bool(const clang::Attr*, SessionStage&)>;
  using HandleType = std::function<bool(const clang::Attr*,
                                        const clang::Stmt*,
                                        SessionStage&,
                                        HandledChanges callback)>;

  AttrStmtHandler(AttrStmtHandler&&) = default;
  AttrStmtHandler(ParamsParserType pp, HandleType h);
  ~AttrStmtHandler() = default;

  bool handle(const clang::Attr* attr,
              const clang::Stmt*,
              SessionStage& stage,
              HandledChanges callback);

 protected:
  bool parseParams(const clang::Attr*, SessionStage& stage);

 private:
  ParamsParserType _paramsParser;
  HandleType _handler;
};
}  // namespace oklt
