#pragma once

#include <oklt/core/error.h>
#include <oklt/core/target_backends.h>

#include "core/rewriter/rewriter_fabric.h"
#include "core/rewriter/rewriter_proxy.h"
#include "core/transpiler_session/header_info.h"

#include <clang/Frontend/CompilerInstance.h>

#include <any>

namespace oklt {

class ASTVisitor;
class HandlerManager;
struct TranspilerSession;

// INFO: could hold not the reference to the global HandlerManager
//       but hold the pointer to the AttributeManagerView
//       that is built for current session with set of interested attribute handlers
class SessionStage {
   public:
    explicit SessionStage(TranspilerSession& session,
                          clang::CompilerInstance& compiler,
                          RewriterProxyType rwType = RewriterProxyType::Original);
    ~SessionStage() = default;

    const TranspilerSession& getSession() const { return _session; }

    TranspilerSession& getSession() { return _session; }

    clang::CompilerInstance& getCompiler();

    oklt::Rewriter& getRewriter();
    std::string getRewriterResultForMainFile();
    TransformedFiles getRewriterResultForHeaders();

    [[nodiscard]] TargetBackend getBackend() const;
    static HandlerManager& getAttrManager();

    void setLauncherMode();

    void pushDiagnosticMessage(clang::StoredDiagnostic& message);
    void pushError(std::error_code ec, std::string desc);
    void pushError(const Error& err);
    void pushWarning(std::string desc);

    inline bool hasUserCtx(const std::string& key) {
        auto it = _userCtxMap.find(key);
        return (it != _userCtxMap.end());
    };
    inline bool setUserCtx(const std::string& key, const std::any& ctx) {
        auto [_, ret] = _userCtxMap.try_emplace(key, ctx);
        return ret;
    }
    inline std::any* getUserCtx(const std::string& key) {
        auto it = _userCtxMap.find(key);
        if (it == _userCtxMap.end())
            return nullptr;

        return &it->second;
    }

    template <typename T, typename... Args>
    inline T& tryEmplaceUserCtx(const std::string& key = typeid(T).name(), Args&&... args) {
        if (!hasUserCtx(key))
            setUserCtx(key, std::make_any<T>(std::forward<Args>(args)...));

        return std::any_cast<T&>(_userCtxMap[key]);
    }

   protected:
    TranspilerSession& _session;
    TargetBackend _backend;
    clang::CompilerInstance& _compiler;
    std::unique_ptr<oklt::Rewriter> _rewriter;
    std::map<std::string, std::any> _userCtxMap;
};

SessionStage* getStageFromASTContext(clang::ASTContext&);

}  // namespace oklt
