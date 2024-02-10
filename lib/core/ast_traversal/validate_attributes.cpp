#include <oklt/core/ast_traversal/validate_attributes.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/pipeline/stages/transpiler/error_codes.h>

namespace oklt {
using namespace clang;

ValidatorResult validateAttributes(const clang::ArrayRef<const clang::Attr*>& attrs,
                                   SessionStage& stage) {
  std::list<const Attr*> collectedAttrs;
  auto& attrManager = stage.getAttrManager();
  for (const auto& attr : attrs) {
    auto name = attr->getNormalizedFullName();
    if (attrManager.hasAttrHandler(attr, stage)) {
      collectedAttrs.push_back(attr);
    }
  }
  if (collectedAttrs.empty()) {
    return nullptr;
  }

  if (collectedAttrs.size() > 1) {
    // TODO: directly push the error to _stage.pushError()

    const Attr* first = collectedAttrs.front();
#if 0
    DiagnosticsEngine &de = stage.getCompiler().getDiagnostics();
    auto id = de.getCustomDiagID(DiagnosticsEngine::Error,
                                 "Multiple OKL attributes are used, total occurenses: %1");
    de.Report(first->getScopeLoc(), id)
      .AddTaggedVal(collectedAttrs.size(),
                    DiagnosticsEngine::ArgumentKind::ak_uint);
#endif
    auto locationStr = first->getScopeLoc().printToString(stage.getCompiler().getSourceManager());
    std::string description = "Location: " + locationStr + ", Multiple OKL attributes are used";
    auto errCode = make_error_code(OkltTranspilerErrorCode::MULTIPLE_ATTRIBUTES_USED);
    stage.pushError(errCode, std::string(description));
    return tl::unexpected<Error>(Error{errCode, description});
  }
  const Attr* attr = collectedAttrs.front();
  return attr;
}
}  // namespace oklt
