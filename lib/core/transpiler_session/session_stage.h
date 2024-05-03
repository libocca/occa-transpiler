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

/**
 * @brief Represents a session of a stage in the normalization/transpilation process.
 *
 * This class holds the state and context for a particular stage of the transpilation process.
 * It provides methods for interacting with the compiler, the rewriter, and the attribute manager,
 * as well as for managing user-defined context data.
 */
class SessionStage {
   public:
    /**
     * @brief Constructs a new SessionStage object.
     *
     * @param session The transpilation session.
     * @param compiler The compiler instance.
     * @param rwType The type of rewriter to use.
     */
    explicit SessionStage(TranspilerSession& session,
                          clang::CompilerInstance& compiler,
                          RewriterProxyType rwType = RewriterProxyType::Original);
    /**
     * @brief Destroys the SessionStage object.
     */
    ~SessionStage() = default;

    /**
     * @brief Gets the transpilation session.
     *
     * @return const TranspilerSession& The transpilation session.
     */
    const TranspilerSession& getSession() const { return _session; }

    /**
     * @brief Gets the transpilation session.
     *
     * @return TranspilerSession& The transpilation session.
     */
    TranspilerSession& getSession() { return _session; }

    /**
     * @brief Gets the compiler instance.
     *
     * @return clang::CompilerInstance& The compiler instance.
     */
    clang::CompilerInstance& getCompiler();

    /**
     * @brief Gets the rewriter.
     *
     * @return oklt::Rewriter& The rewriter.
     */
    [[nodiscard]] oklt::Rewriter& getRewriter();

    /**
     * @brief Gets the results of all the rewrites for the main file.
     *
     * @return std::string The rewriter result for the main file.
     */
    std::string getRewriterResultForMainFile();

    /**
     * @brief Gets the results of all the rewrites for the header files.
     *
     * @return TransformedFiles The rewriter result for the header files.
     */
    TransformedFiles getRewriterResultForHeaders();

    /**
     * @brief Gets the target backend.
     *
     * @return TargetBackend The target backend.
     */
    [[nodiscard]] TargetBackend getBackend() const;

    /**
     * @brief Gets the attribute manager.
     *
     * @return HandlerManager& The attribute manager.
     */
    [[nodiscard]] HandlerManager& getAttrManager();

    /**
     * @brief Sets the launcher mode.
     */
    void setLauncherMode();

    /**
     * @brief Save a diagnostic message.
     *
     * @param message The diagnostic message to push.
     */
    void pushDiagnosticMessage(clang::StoredDiagnostic& message);

    /**
     * @brief Add error.
     *
     * @param ec The error code.
     * @param desc The error description.
     */
    void pushError(std::error_code ec, std::string desc);

    /**
     * @brief Add error.
     *
     * @param err The error to push.
     */
    void pushError(const Error& err);

    /**
     * @brief Add warning.
     *
     * @param desc The warning description.
     */
    void pushWarning(std::string desc);

    /**
     * @brief Checks if a user context exists.
     *
     * @param key The key of the user context to check.
     * @return bool True if the user context exists, false otherwise.
     */
    inline bool hasUserCtx(const std::string& key) {
        auto it = _userCtxMap.find(key);
        return (it != _userCtxMap.end());
    };
    /**
     * @brief Sets a user context.
     *
     * @param key The key of the user context to set.
     * @param ctx The user context to set.
     * @return bool True if the user context was set successfully, false otherwise.
     */

    inline bool setUserCtx(const std::string& key, const std::any& ctx) {
        auto [_, ret] = _userCtxMap.try_emplace(key, ctx);
        return ret;
    }
    /**
     * @brief Gets a user context.
     *
     * @param key The key of the user context to get.
     * @return std::any* A pointer to the user context, or nullptr if the user context does not
     * exist.
     */
    inline std::any* getUserCtx(const std::string& key) {
        auto it = _userCtxMap.find(key);
        if (it == _userCtxMap.end())
            return nullptr;

        return &it->second;
    }

    /**
     * @brief Tries to emplace a user context.
     *
     * @tparam T The type of the user context to emplace.
     * @tparam Args The types of the arguments to forward to the constructor of the user context.
     * @param key The key of the user context to emplace.
     * @param args The arguments to forward to the constructor of the user context.
     * @return T& A reference to the emplaced user context.
     */
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

/**
 * @brief Gets the session stage from an AST context.
 *
 * @param context The clang AST context.
 * @return SessionStage* A pointer to the session stage.
 */
SessionStage* getStageFromASTContext(clang::ASTContext&);

}  // namespace oklt
