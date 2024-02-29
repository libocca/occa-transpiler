#pragma once

#include <oklt/core/ast_processor_types.h>
#include <oklt/core/error.h>
#include <oklt/core/target_backends.h>

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include <any>

namespace oklt {

class ASTVisitor;
class AttributeManager;
struct TranspilerSession;

// INFO: could hold not the reference to the global AttributeManager
//       but hold the pointer to the AttributeManagerView
//       that is built for current session with set of interested attribute handlers
class SessionStage {
   public:
    explicit SessionStage(TranspilerSession& session, clang::CompilerInstance& compiler);
    ~SessionStage() = default;

    const TranspilerSession& getSession() const { return _session; }

    TranspilerSession& getSession() { return _session; }

    clang::CompilerInstance& getCompiler();

    clang::Rewriter& getRewriter();
    std::string getRewriterResult();

    [[nodiscard]] TargetBackend getBackend() const;
    [[nodiscard]] AstProcessorType getAstProccesorType() const;
    static AttributeManager& getAttrManager();

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

    clang::CompilerInstance& _compiler;
    clang::Rewriter _rewriter;

    // XXX discuss key
    std::map<std::string, std::any> _userCtxMap;
};

SessionStage* getStageFromASTContext(clang::ASTContext&);

}  // namespace oklt
