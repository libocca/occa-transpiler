#include "oklt/core/ast_traversal/semantic_analyzer.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/attribute_manager/attributed_type_map.h"

namespace oklt {
using namespace clang;


template<typename ... Ts>
struct Overload : Ts ... {
  using Ts::operator() ...;
};
template<class... Ts> Overload(Ts...) -> Overload<Ts...>;

//INFO: simple mapping
inline DatatypeCategory makeDatatypeCategory(const QualType &qt) {
  if(qt->isBuiltinType()) {
    return DatatypeCategory::BUILTIN;
  }
  return DatatypeCategory::CUSTOM;
}

SemanticAnalyzer::SemanticAnalyzer(SEMANTIC_CATEGORY category,
                                   SessionStage& stage)
    : _category(category)
    , _stage(stage)
    , _kernels()
    , _astKernels()

{}

SemanticAnalyzer::KernelInfoT& SemanticAnalyzer::getKernelInfo() {
  return _kernels;
}

bool SemanticAnalyzer::TraverseDecl(clang::Decl* decl) {
  return RecursiveASTVisitor<SemanticAnalyzer>::TraverseDecl(decl);
}

bool SemanticAnalyzer::TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue) {
  return RecursiveASTVisitor<SemanticAnalyzer>::TraverseStmt(stmt, queue);
}

bool SemanticAnalyzer::TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue)
{
  auto subExpr = expr->subExpressions();
  if (subExpr.empty()) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseRecoveryExpr(expr, queue);
  }

  auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
  if (!declRefExpr) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseRecoveryExpr(expr, queue);
  }

  auto& ctx = _stage.getCompiler().getASTContext();
  auto& attrTypeMap = _stage.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto validationHandlers = Overload {
    [this, expr](const clang::Attr*attr) -> bool {
      auto& attrManager = _stage.getAttrManager();
      if (!attrManager.handleAttr(attr, expr, _stage, nullptr)) {
        return false;
      }
      return true;
    },
    [this](const NoOKLAttrs&) -> bool {

      return true;
    },
    [this](const ErrorFired& error) -> bool {
      return false;
    },
  };

  auto validationResult = validateAttribute(attrs);
  return std::visit(validationHandlers, validationResult);
}

bool SemanticAnalyzer::TraverseFunctionDecl(clang::FunctionDecl *funcDecl)
{
  if(!funcDecl->hasAttrs()) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseFunctionDecl(funcDecl);
  }
  auto &attrs = funcDecl->getAttrs();
  auto validationResult = validateAttribute(attrs);

  auto attrHandleFunc =
    [this, funcDecl](const clang::Attr*attr) -> bool {
    auto& attrManager = _stage.getAttrManager();
    if(_category == SEMANTIC_CATEGORY::HOST_KERNEL_CATEGORY) {
      if (!attrManager.handleAttr(attr, funcDecl, _stage, nullptr)) {
        return false;
      }
      _astKernels.push_back(KernelASTInfo { funcDecl, {}});
      std::vector<ArgumentInfo> args;
      for(const auto &param : funcDecl->parameters()) {
        auto paramQualType = param->getType();
        auto typeInfo = param->getASTContext().getTypeInfo(paramQualType);
        args.push_back(ArgumentInfo {
          .is_const = paramQualType.isConstQualified(),
          .dtype = DataType {
            .name = paramQualType.getAsString(),
            .type = makeDatatypeCategory(paramQualType),
            .bytes = static_cast<int>(typeInfo.Width),
          },
          .name = param->getNameAsString(),
          .is_ptr = paramQualType->isPointerType(),
        });
      }
      ParsedKernelInfo info {
        .arguments = std::move(args),
        .name = funcDecl->getNameAsString()
      };
      _kernels.push_back(std::move(info));
      return true;
    } else {
      std::string functionSignature;
      auto changesHandler = [&functionSignature](const Changes &changes) {
        if(changes.empty()) {
          //TODO: INTERNAL error
        }
        functionSignature = changes[0].to;
      };
      if (!attrManager.handleAttr(attr, funcDecl, _stage, changesHandler)) {
        return false;
      }
      functionSignature += " " + funcDecl->getNameAsString();
      _astKernels.push_back(KernelASTInfo { funcDecl, {}});
      return true;
    }
  };

  auto validationHandlers = Overload {
    attrHandleFunc,
    [this, funcDecl](const NoOKLAttrs&) -> bool{
      return RecursiveASTVisitor<SemanticAnalyzer>::TraverseDecl(funcDecl);
    },
    [](const ErrorFired&) -> bool {
      return false;
    },
  };
  return std::visit(validationHandlers, validationResult);
}

bool SemanticAnalyzer::TraverseAttributedStmt(clang::AttributedStmt *attrStmt,
                                              DataRecursionQueue* queue)
{
  auto attrs = attrStmt->getAttrs();
  auto validationResult = validateAttribute(attrs);
  auto validationHandlers = Overload {
    [this, attrStmt, queue](const clang::Attr*attr) -> bool {
      auto& attrManager = _stage.getAttrManager();
      const Stmt* subStmt = attrStmt->getSubStmt();
      if (!attrManager.handleAttr(attr, subStmt, _stage, nullptr)) {
        return false;
      }
      return true;
    },
    [this, attrStmt, queue](const NoOKLAttrs&) -> bool{
      return RecursiveASTVisitor<SemanticAnalyzer>::TraverseAttributedStmt(attrStmt, queue);
    },
    [](const ErrorFired&) -> bool {
      return false;
    },
  };
  return std::visit(validationHandlers, validationResult);
}

SemanticAnalyzer::ValidationResult SemanticAnalyzer::validateAttribute(const clang::ArrayRef<const clang::Attr *> &attrs)
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
    return SemanticAnalyzer::NoOKLAttrs{};
  }

  if (collectedAttrs.size() > 1) {
    const Attr* first = collectedAttrs.front();
    DiagnosticsEngine &de = _stage.getCompiler().getDiagnostics();
    auto id = de.getCustomDiagID(DiagnosticsEngine::Error, "Multiple OKL attributes are used, total occurenses: %1");
    de.Report(first->getScopeLoc(), id)
      .AddTaggedVal(collectedAttrs.size(),
                    DiagnosticsEngine::ArgumentKind::ak_uint);
    return ErrorFired {};
  }
  const Attr* attr = collectedAttrs.front();
  return attr;
}

}
