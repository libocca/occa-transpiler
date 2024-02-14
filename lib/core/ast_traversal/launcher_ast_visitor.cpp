#include "oklt/core/ast_traversal/launcher_ast_visitor.h"
#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/attribute_manager/attributed_type_map.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

#include <clang/AST/ParentMapContext.h>

namespace oklt {
using namespace clang;

namespace {

template <typename T>
static T* GetAttr(Decl* D, llvm::StringRef attrName) {
  for (auto attr : D->getAttrs()) {
    if (attr && isa<T>(attr) && attr->getNormalizedFullName() == attrName) {
      return static_cast<T*>(attr);
    }
  }

  return nullptr;
};

template <typename T>
static T* GetAttr(AttributedStmt* S, llvm::StringRef attrName) {
  for (auto attr : S->getAttrs()) {
    if (attr && isa<T>(attr) && attr->getNormalizedFullName() == attrName) {
      return static_cast<T*>(const_cast<Attr*>(attr));
    }
  }

  return nullptr;
};

std::string prettyPrint(Stmt* S, const PrintingPolicy& policy) {
  std::string ret = "";
  if (!S) {
    return ret;
  }

  llvm::raw_string_ostream os(ret);
  S->printPretty(os, nullptr, policy);

  return ret;
};

struct LoopMetadata {
  std::string type;
  std::string name;
  struct {
    std::string start;
    std::string end;
    std::string size_str;
    size_t size = 0;
  } range;
  struct {
    std::string orig;
    std::string cmp;
    BinaryOperator::Opcode op = BO_EQ;
  } condition;
  struct {
    std::string val;
    union {
      UnaryOperator::Opcode uo;
      BinaryOperator::Opcode bo;
    } op;
  } inc;
};

struct ParamMetadata {
  std::string type = {};
  std::string name = {};
  ParamMetadata(const std::string& t, const std::string& n) : type(t), name(n){};
};

struct KernelInstanceMetadata {
  std::list<LoopMetadata> outer = {};
  std::list<LoopMetadata> inner = {};
};

struct LKernelMetadata {
  std::string name = {};
  std::list<ParamMetadata> params = {};
  std::list<KernelInstanceMetadata> instances = {};
};

};

class LauncherKernelGenerator {
 public:
  LauncherKernelGenerator(SessionStage& stage): _ctx(stage.getCompiler().getASTContext()), _policy(_ctx.getPrintingPolicy()) {};

  void ParseFunctionDecl(FunctionDecl* D, AnnotateAttr& Attr) {
    if (!D || !D->hasBody())
      return;

    _metadata.name = D->getName();
    _metadata.params.emplace_back("occa::modeKernel_t **", "deviceKernels");
    for (auto p: D->parameters()) {
      ParseParmValDecl(p);
    }
  }

  void ParseParmValDecl(ParmVarDecl* D) {
    if (!D)
      return;

    auto t = D->getType();
    if (!t.getNonReferenceType()->isFundamentalType()) {
      _metadata.params.emplace_back("occa::modeMemory_t *", D->getNameAsString());
    } else {
      _metadata.params.emplace_back(t.getNonReferenceType().getAsString() + " &",
                                    D->getNameAsString());
    }
  }

  void ParseTiledForStmt(ForStmt* S, SuppressAttr& Attr) {
    if (!S)
      return;

    const std::string p = "_occa_tiled_";

    // Assume: @tile(16, @outer, @inner)
    const size_t tile_size = 16;
    const bool tile_outer = true;
    const bool tile_inner = true;

    InitInstance(S);
    auto loop_meta = ParseForStmt(S);

    // Prepare outer loop
    LoopMetadata outer = loop_meta;
    outer.name = p + outer.name;
    if (tile_size > 1) {
      if (outer.inc.val.empty()) {
        outer.inc.val = std::to_string(tile_size);
        switch (outer.inc.op.uo) {
          case UO_PreInc:
          case UO_PostInc:
            outer.inc.op.bo = BO_AddAssign;
            break;
          case UO_PreDec:
          case UO_PostDec:
            outer.inc.op.bo = BO_RemAssign;
            break;
        }
      } else {
        outer.inc.val = "( " + std::to_string(tile_size) + " * " + outer.inc.val + " )";
      }
    }
    _metadata.instances.back().outer.push_back(outer);

    // Prepare inner loop
    LoopMetadata inner = loop_meta;
    inner.range.start = outer.name;
    switch (inner.condition.op) {
      case BO_LE:
        inner.condition.op = BO_LT;
        break;
      case BO_GE:
        inner.condition.op = BO_GT;
        break;
    }
    if (tile_size > 1) {
      inner.condition.cmp = "( " + outer.name + " + " + std::to_string(tile_size) + " )";
    } else {
      inner.condition.cmp = outer.name;
    }
    _metadata.instances.back().inner.push_back(inner);
  }

  void ParseOuterForStmt(ForStmt* S, SuppressAttr& Attr) {
    if (!S)
      return;

    InitInstance(S);
    _metadata.instances.back().outer.emplace_back(ParseForStmt(S));
  }

  void ParseInnerForStmt(ForStmt* S, SuppressAttr& Attr) {
    if (!S)
      return;

    _metadata.instances.back().inner.emplace_back(ParseForStmt(S));
  }

  std::string GenerateSource() {
    std::stringstream out;

    unsigned i = 0; // ident
    auto getIdent = [](unsigned v) -> std::string {
      return std::string(v * 2, ' ');
    };

    // Function name
    out << "\nextern \"C\" void " << _metadata.name;

    // Parameters
    out << "( ";
    for (auto it = _metadata.params.begin(), end_it = _metadata.params.end(); it != end_it; ++it) {
      if (it != _metadata.params.begin())
        out << ", ";

      auto &p = *it;
      out << p.type << " " << p.name;
    }
    out << " ) {\n";

    ++i;
    auto k = size_t(0);
    for (auto &instance : _metadata.instances) {
      out << getIdent(i++) << "{\n";

      auto s = getIdent(i);
      out << s << "occa::dim outer, inner;\n";
      out << s << "outer.dims = " << instance.outer.size() << ";\n";
      out << s << "inner.dims = " << instance.outer.size() << ";\n";

      auto format_loop = [&out, &s](const LoopMetadata& loop, size_t n, bool isOuter) -> void {
        out << s << loop.type << " " << loop.name << " = " << loop.range.start << ";\n";
        out << s << (isOuter ? "outer" : "inner") << "[" << n << "] = ";

        if (!loop.inc.val.empty())
          out << "( ";

        switch (loop.condition.op) {
          case BO_LE:
          case BO_GE:
            out << "1 + ";
            break;
        }

        out << loop.range.size_str;

        if (!loop.inc.val.empty()) {
          out << " + " << loop.inc.val << " - 1) / " << loop.inc.val;
        }

        out << ";\n";
      };

      auto n = instance.outer.size();
      for (auto &loop : instance.outer) {
        format_loop(loop, n, true);
        --n;
      }

      n = instance.inner.size();
      for (auto &loop : instance.inner) {
        format_loop(loop, n, false);
        --n;
      }

      out << s << "occa::kernel kernel(deviceKernels[" << k << "]);\n";
      out << s << "kernel.setRunDims(outer, inner);\n";

      out << s << "kernel";
      out << "( ";
      for (auto it = _metadata.params.begin(), end_it = _metadata.params.end(); it != end_it; ++it) {
        if (it != _metadata.params.begin())
          out << ", ";

        out << it->name;
      }
      out << " );\n";

      out << getIdent(--i) << "}" << std::endl;
    }

    out << "}\n";
    out.flush();

    return out.str();
  }

 protected:
  ASTContext &_ctx;
  const PrintingPolicy& _policy;
  LKernelMetadata _metadata = {};

  template<typename ParentT, typename NodeT = ParentT>
  [[nodiscard]] inline ParentT *getParent(NodeT *Node) {
    if (!Node)
      return nullptr;

    const auto Parents = _ctx.getParentMapContext().getParents(*Node);
    if (Parents.empty())
      return nullptr;

    return const_cast<ParentT *>(Parents[0].template get<ParentT>());
  };

  void InitInstance(Stmt* S) {
    auto attr_stmt = getParent<AttributedStmt>(S);
    auto comp_stmt = getParent<CompoundStmt>(attr_stmt);
    auto func_decl = getParent<FunctionDecl>(comp_stmt);
    if (!func_decl) {
      return;
    }

    _metadata.instances.emplace_back();
  };

  bool EvaluateAsSizeT(const Expr *E, llvm::APSInt &Into) {
    unsigned BitsInSizeT = _ctx.getTypeSize(_ctx.getSizeType());

    Expr::EvalResult ExprResult;
    if (!E->EvaluateAsInt(ExprResult, _ctx, Expr::SE_AllowSideEffects))
      return false;
    Into = ExprResult.Val.getInt();
    if (Into.isNegative() || !Into.isIntN(BitsInSizeT))
      return false;
    Into = Into.zext(BitsInSizeT);
    return true;
  };

  LoopMetadata ParseForStmt(ForStmt* S) {
    LoopMetadata ret;
    Expr *start, *end;

    if (isa<DeclStmt>(S->getInit())) {
      auto d = dyn_cast<DeclStmt>(S->getInit());
      if (!d->isSingleDecl()) {
        // TODO: throw Multi-Declaration
        return ret;
      }

      auto node = dyn_cast<VarDecl>(d->getSingleDecl());
      if (!node) {
        // TODO: throw No Init-statement
        return ret;
      }

      ret.name = node->getDeclName().getAsString();
      ret.type = node->getType().getAsString();

      start = node->getInit();
      ret.range.start = prettyPrint(start, _policy);
    }

    // Condition
    if (isa<BinaryOperator>(S->getCond())) {
      auto node = dyn_cast<BinaryOperator>(S->getCond());
      if (!node->isComparisonOp()) {
        // TODO: throw Non Comparison OP
        return ret;
      }

      ret.condition.op = node->getOpcode();
      ret.condition.orig = prettyPrint(node, _policy);

      auto lsh = dyn_cast_or_null<ImplicitCastExpr>(node->getLHS());
      auto decl = dyn_cast_or_null<DeclRefExpr>(lsh ? lsh->getSubExpr() : nullptr);
      if (decl && decl->getNameInfo().getAsString() == ret.name) {
        end = node->getRHS();
        ret.range.end = prettyPrint(end, _policy);
      }

      auto rsh = dyn_cast_or_null<ImplicitCastExpr>(node->getRHS());
      decl = dyn_cast_or_null<DeclRefExpr>(rsh ? rsh->getSubExpr() : nullptr);
      if (decl && decl->getNameInfo().getAsString() == ret.name) {
        end = node->getLHS();
        ret.range.end = prettyPrint(end, _policy);
        ret.condition.op = BinaryOperator::reverseComparisonOp(node->getOpcode());
      }

      if (!end) {
        // TODO: throw Condition not using init variable
        return ret;
      }
      ret.condition.cmp = prettyPrint(end, _policy);
    }

    bool is_inc = false;
    // Increment
    if (isa<UnaryOperator>(S->getInc())) {
      auto node = dyn_cast<UnaryOperator>(S->getInc());
      ret.inc.op.uo = node->getOpcode();

      const auto inc_op = ret.inc.op.uo;
      const auto cmp_op = ret.condition.op;
      is_inc = ((inc_op == UO_PreInc || inc_op == UO_PostInc) && (cmp_op == BO_LE || cmp_op == BO_LT));
    }

    if (isa<CompoundAssignOperator>(S->getInc())) {
      auto node = dyn_cast<CompoundAssignOperator>(S->getInc());

      auto lsh = dyn_cast_or_null<DeclRefExpr>(node->getLHS());
      if (lsh && lsh->getNameInfo().getAsString() != ret.name) {
        // TODO: throw Declaration is not incremented?
        return ret;
      }

      ret.inc.op.bo = node->getOpcode();
      ret.inc.val = prettyPrint(node->getRHS(), _policy);

      const auto inc_op = ret.inc.op.bo;
      const auto cmp_op = ret.condition.op;
      is_inc = (inc_op == BO_AddAssign && (cmp_op == BO_LE || cmp_op == BO_LT));
    }

    ret.range.size = 0;

    // Determinate range size
    llvm::APSInt start_i, end_i;
    if (EvaluateAsSizeT(start, start_i) && EvaluateAsSizeT(end, end_i)) {
      if (is_inc) {
        end_i -= start_i;
        ret.range.size = end_i.getZExtValue();
      } else {
        start_i -= end_i;
        ret.range.size = start_i.getZExtValue();
      }
    }

    if (is_inc) {
      ret.range.size_str = ret.range.end + " - " + ret.range.start;
    } else {
      ret.range.size_str = ret.range.start + " - " + ret.range.end;
    }

    return ret;
  }

};

LauncherASTVisitor::LauncherASTVisitor(SessionStage& stage): _stage(stage) {};

bool LauncherASTVisitor::TraverseTranslationUnitDecl(TranslationUnitDecl* D) {
  switch (_stage.getBackend()) {
    //case TargetBackend::SERIAL:
    case TargetBackend::OPENMP:
      return true;
    default:
      _source << "#include <occa/core/kernel.hpp>\n\n";
      break;
  }

  if (!Base::TraverseTranslationUnitDecl(D)) {
    return false;
  }

  _stage.setUserCtx("launcher", _source.str());
  llvm::outs() << _source.str();

  return true;
}

bool LauncherASTVisitor::TraverseFunctionDecl(FunctionDecl* D) {
  if (!D || !D->hasBody() || !D->hasAttrs()) {
    return Base::TraverseFunctionDecl(D);
  }

  auto kernelAttr = GetAttr<AnnotateAttr>(D, "okl::kernel");
  if (!kernelAttr) {
    return Base::TraverseFunctionDecl(D);
  }

  auto generator = std::make_unique<LauncherKernelGenerator>(_stage);
  _generator = generator.get();

  generator->ParseFunctionDecl(D, *kernelAttr);

  auto ret = Base::TraverseFunctionDecl(D);

  _source << _generator->GenerateSource();

  _generator = nullptr;
  generator.reset();

  return ret;
}

bool LauncherASTVisitor::TraverseAttributedStmt(AttributedStmt* S, DataRecursionQueue* Queue) {
  if (!S) {
    return false;
  }

  auto forStmt = dyn_cast<ForStmt>(S->getSubStmt());
  if (!forStmt) {
    return false;
  }

  if (!_generator) {
    return Base::TraverseAttributedStmt(S, Queue);
  }

  if (auto tileAttr = GetAttr<SuppressAttr>(S, "okl::tile")) {
    _generator->ParseTiledForStmt(forStmt, *tileAttr);
  } else if (auto outerAttr = GetAttr<SuppressAttr>(S, "okl::outer")) {
    _generator->ParseOuterForStmt(forStmt, *outerAttr);
  } else if (auto innerAttr = GetAttr<SuppressAttr>(S, "okl::inner")) {
    _generator->ParseInnerForStmt(forStmt, *innerAttr);
  }

  return Base::TraverseAttributedStmt(S, Queue);
}

std::unique_ptr<LauncherASTVisitor> LauncherASTVisitor::Create(SessionStage& stage) {
  return std::make_unique<LauncherASTVisitor>(stage);
}

}  // namespace oklt
