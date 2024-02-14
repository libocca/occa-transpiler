#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;
using namespace clang;

bool parseSharedAttribute(const Attr* a, SessionStage&) {
  llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
  return true;
}

bool handleSharedAttribute(const Attr* a, const Decl* d, SessionStage& s) {
  llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
  auto& rewriter = s.getRewriter();

  SourceRange range = a->getRange();
  SourceLocation begin = range.getBegin();
  int length = rewriter.getRangeSize(range);
  if (length == -1) {
    // TODO: internal error
    return false;
  }
  if (a->isCXX11Attribute() || a->isC2xAttribute()) {
    begin = begin.getLocWithOffset(-2);
    length += 4;
  }
  if (a->isGNUAttribute()) {
    begin = begin.getLocWithOffset(-15);
    length += 17;
  }
  rewriter.RemoveText(begin, length);
  std::string sharedText = "__shared__ ";
  return rewriter.InsertText(d->getBeginLoc(), sharedText, false, false);
}

__attribute__((constructor)) void registerSharedHandler() {
  auto ok = oklt::AttributeManager::instance().registerBackendHandler(
    {TargetBackend::CUDA, SHARED_ATTR_NAME}, {parseSharedAttribute, handleSharedAttribute});

  if (!ok) {
    llvm::errs() << "failed to register " << SHARED_ATTR_NAME << " attribute handler\n";
  }
}
}  // namespace
