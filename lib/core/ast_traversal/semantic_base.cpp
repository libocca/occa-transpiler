#include <oklt/core/ast_traversal/semantic_base.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/attribute_manager/attribute_manager.h>

namespace oklt {
using namespace clang;

SemanticASTVisitorBase::SemanticASTVisitorBase(SessionStage& stage)
    :_stage(stage)
{}

SemanticASTVisitorBase::~SemanticASTVisitorBase()
{}

SemanticASTVisitorBase::ValidationResult SemanticASTVisitorBase::validateAttribute(
  const clang::ArrayRef<const clang::Attr *> &attrs)
{
  std::list<const Attr*> collectedAttrs;
  auto &attrManager = _stage.getAttrManager();
  for (const auto& attr : attrs) {
    auto name = attr->getNormalizedFullName();
    if(attrManager.hasAttrHandler(attr, _stage)) {
      collectedAttrs.push_back(attr);
    }
  }
  if (collectedAttrs.empty()) {
    return NoOKLAttrs{};
  }

  if (collectedAttrs.size() > 1) {
    const Attr* first = collectedAttrs.front();
    DiagnosticsEngine &de = _stage.getCompiler().getDiagnostics();
    auto id = de.getCustomDiagID(DiagnosticsEngine::Error,
                                 "Multiple OKL attributes are used, total occurenses: %1");
    de.Report(first->getScopeLoc(), id)
      .AddTaggedVal(collectedAttrs.size(),
                    DiagnosticsEngine::ArgumentKind::ak_uint);
    return ErrorFired {};
  }
  const Attr* attr = collectedAttrs.front();
  return attr;
}

}
