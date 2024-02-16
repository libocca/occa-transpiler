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

std::string_view noParen(const std::string& s) {
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')')
    return { s.data() + 1, s.size() - 2 };
  return s;
}

struct LoopMetadata {
  std::string type;
  std::string name;
  struct {
    std::string start;
    std::string end;
    size_t size = 0;
  } range;
  struct {
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

  bool IsInc() const {
    bool ret = false;
    if (inc.val.empty()) {
      ret = (inc.op.uo == UO_PreInc || inc.op.uo == UO_PostInc);
    } else {
      ret = (inc.op.bo == BO_AddAssign);
    }
    ret = (ret && (condition.op == BO_LE || condition.op == BO_LT));

    return ret;
  };
  std::string getRangeSizeStr() const {
    if (IsInc()) {
      return range.end + " - " + range.start;
    } else {
      return range.start + " - " + range.end;
    };
  };
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

    // TODO: Add Attr parser somewhere else?
    auto parseTileAttr = [](SuppressAttr& Attr) -> std::tuple<size_t, bool, bool> {
      auto str = StringRef(*Attr.diagnosticIdentifiers_begin());
      if (str.starts_with("(") && str.ends_with(")"))
        str = str.substr(1, str.size() - 2);
      auto [sz_str, rsh] = str.split(',');
      return { std::atoi(sz_str.data()), rsh.contains("@outer"), rsh.contains("@inner") };
    };
    const auto [tile_size, tile_outer, tile_inner] = parseTileAttr(Attr);

    InitInstance(S);

    if (_metadata.instances.empty())
      return;
    auto &instance = _metadata.instances.back();

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
        outer.inc.val = "(" + std::to_string(tile_size) + " * " + outer.inc.val + ")";
      }
    }
    if (tile_outer)
      instance.outer.push_back(outer);

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
      inner.range.end = "(" + outer.name + " + " + std::to_string(tile_size) + ")";
    } else {
      inner.range.end = outer.name;
    }
    if (tile_inner)
      instance.inner.push_back(inner);
  }

  void ParseOuterForStmt(ForStmt* S, SuppressAttr& Attr) {
    if (!S)
      return;

    InitInstance(S);

    if (_metadata.instances.empty())
      return;
    auto &instance = _metadata.instances.back();

    instance.outer.emplace_back(ParseForStmt(S));
  }

  void ParseInnerForStmt(ForStmt* S, SuppressAttr& Attr) {
    if (!S)
      return;

    if (_metadata.instances.empty())
      return;

    auto &instance = _metadata.instances.back();
    instance.inner.emplace_back(ParseForStmt(S));
  }

  std::string GenerateSource() {
    std::stringstream out;

    unsigned i = 0; // ident
    auto getIdent = [](unsigned v) -> std::string {
      return std::string(v * 2, ' ');
    };

    // Function
    out << "extern \"C\" void " << _metadata.name;
    out << "(";
    {
      const auto param_ident = std::string(15 + _metadata.name.size() + 2, ' ');
      for (auto it = _metadata.params.begin(), end_it = _metadata.params.end(); it != end_it;
           ++it) {
        if (it != _metadata.params.begin())
          out << ",\n" << param_ident;

        auto& p = *it;
        out << p.type << " " << p.name;
      }
    }
    out << ") {\n";

    ++i;
    auto k = size_t(0);
    for (auto &instance : _metadata.instances) {
      out << getIdent(i++) << "{\n";

      auto s = getIdent(i);
      out << s << "occa::dim outer, inner;\n";
      out << s << "outer.dims = " << instance.outer.size() << ";\n";
      out << s << "inner.dims = " << instance.inner.size() << ";\n";

      auto format_loop = [&out, &s](const LoopMetadata& loop, size_t n, bool isOuter) -> void {
        out << s << loop.type << " " << loop.name << " = " << noParen(loop.range.start) << ";\n";
        out << s << (isOuter ? "outer" : "inner") << "[" << n << "] = ";

        if (!loop.inc.val.empty())
          out << "(";

        switch (loop.condition.op) {
          case BO_LE:
          case BO_GE:
            out << "1 + ";
            break;
        }

        out << loop.getRangeSizeStr();

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
      out << "(";
      {
        bool is_first = true;
        for (auto it = std::next(_metadata.params.begin()), end_it = _metadata.params.end();
             it != end_it; ++it) {
          if (!is_first)
            out << ", ";
          is_first = false;

          out << it->name;
        }
      }
      out << ");\n";

      out << getIdent(--i) << "}\n";
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
      while (auto rsh = dyn_cast_or_null<CastExpr>(start)) {
        start = rsh->getSubExpr();
      }
      ret.range.start = prettyPrint(start, _policy);

      auto child_count = std::distance(start->children().begin(), start->children().end());
      if (child_count > 0 && !node->getInit()->isEvaluatable(_ctx)) {
        ret.range.start = "(" + ret.range.start + ")";
      }
    }

    // Condition
    if (isa<BinaryOperator>(S->getCond())) {
      auto node = dyn_cast<BinaryOperator>(S->getCond());
      if (!node->isComparisonOp()) {
        // TODO: throw Non Comparison OP
        return ret;
      }

      ret.condition.op = node->getOpcode();
      ret.condition.cmp = prettyPrint(node, _policy);

      // LSH
      auto lsh = dyn_cast_or_null<CastExpr>(node->getLHS());
      while (lsh && lsh->getSubExpr() && isa<CastExpr>(lsh->getSubExpr())) {
        lsh = dyn_cast_or_null<CastExpr>(lsh->getSubExpr());
      };
      if (lsh && lsh->getSubExpr()) {
        auto decl = dyn_cast_or_null<DeclRefExpr>(lsh->getSubExpr());
        if (decl && decl->getNameInfo().getAsString() == ret.name) {
          end = node->getRHS();
          ret.range.end = prettyPrint(end, _policy);
        }
      };

      // RSH
      auto rsh = dyn_cast_or_null<CastExpr>(node->getRHS());
      while (rsh && rsh->getSubExpr() && isa<CastExpr>(rsh->getSubExpr())) {
        rsh = dyn_cast_or_null<CastExpr>(rsh->getSubExpr());
      };
      if (rsh && rsh->getSubExpr()) {
        auto decl = dyn_cast_or_null<DeclRefExpr>(rsh->getSubExpr());
        if (decl && decl->getNameInfo().getAsString() == ret.name) {
          end = node->getLHS();
          ret.range.end = prettyPrint(end, _policy);
          ret.condition.op = BinaryOperator::reverseComparisonOp(node->getOpcode());
        }
      }

      if (!end) {
        // TODO: throw Condition not using init variable
        return ret;
      }
    }

    // Increment
    if (isa<UnaryOperator>(S->getInc())) {
      auto node = dyn_cast<UnaryOperator>(S->getInc());
      ret.inc.op.uo = node->getOpcode();
    } else if (isa<CompoundAssignOperator>(S->getInc())) {
      auto node = dyn_cast<CompoundAssignOperator>(S->getInc());

      auto lsh = dyn_cast_or_null<DeclRefExpr>(node->getLHS());
      if (lsh && lsh->getNameInfo().getAsString() != ret.name) {
        // TODO: throw Declaration is not incremented?
        return ret;
      }

      ret.inc.op.bo = node->getOpcode();
      ret.inc.val = prettyPrint(node->getRHS(), _policy);
    }

    ret.range.size = 0;

    // Determinate range size
    llvm::APSInt start_i, end_i;
    if (EvaluateAsSizeT(start, start_i) && EvaluateAsSizeT(end, end_i)) {
      if (ret.IsInc()) {
        end_i -= start_i;
        ret.range.size = end_i.getZExtValue();
      } else {
        start_i -= end_i;
        ret.range.size = start_i.getZExtValue();
      }
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

  _source << "\n" << _generator->GenerateSource();

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
