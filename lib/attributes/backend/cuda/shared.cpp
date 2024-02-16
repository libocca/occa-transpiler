#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/utils/attributes.h>

namespace {
using namespace oklt;
using namespace clang;

//bool parseSharedAttribute(const Attr* a, SessionStage&) {
//#ifdef TRANSPILER_DEBUG_LOG
//  llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
//#endif
//  return true;
//}

bool handleSharedAttribute(const Attr* a, const Decl* d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
  llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
  auto& rewriter = s.getRewriter();
  removeAttribute(a, s);

  std::string sharedText = "__shared__ ";
  return rewriter.InsertText(d->getBeginLoc(), sharedText, false, false);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, SHARED_ATTR_NAME}, AttrDeclHandler{handleSharedAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
