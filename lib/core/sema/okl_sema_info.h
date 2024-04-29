#pragma once

#include <oklt/core/kernel_metadata.h>
#include "attributes/frontend/params/loop.h"

#include <clang/AST/AST.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Type.h>

#include <optional>

namespace oklt {

struct KernelInfo;

using LoopTypes = std::vector<LoopType>;
// I know that plural of Axis is Axes, but that way it isn't similar to lumberjack Axe
using Axises = std::vector<Axis>;

/**
 * @struct OklLoopInfo
 * @brief This structure represents information about an loop in the OKL kernel.
 *
 * It contains information about the loop's attributes, statement, type, axis, parent, children,
 * tile size, shared and exclusive info, overridden inner sizes, variable, range, condition, and
 * increment.
 */
struct OklLoopInfo {
    /**
     * @struct AttributedTypeInfo
     * @brief This structure represents some meta information about an attributed type.
     *
     * It contains information about whether the attributed type is declared in this loop (applied
     * to @outer only) and whether the attributed type is used in this or child loops (applied to
     * @inner only).
     */
    struct AttributedTypeInfo {
        bool declared = false;
        bool used = false;
    };

    using OptSize = std::optional<size_t>;

    /**
     * @struct OptSizes
     * @brief This structure represents sizes of each axis in the @inner loops. If loop size is not
     * known at compile time, it is represented as nullopt.
     *
     * It contains methods for getting the product of the sizes, checking if it has null options,
     * and checking if all options are null.
     */
    struct OptSizes : public std::array<OptSize, N_AXIS> {
        /**
         * @brief product of all sizes. nullopt size is equivalent to 1.
         *
         * @return size_t
         */
        size_t product();

        /**
         * @brief Check if any of the sizes is nullopt
         *
         * @return true
         * @return false
         */
        bool hasNullOpts();
        /**
         * @brief Check if all sizes are nullopt
         *
         * @return true
         * @return false
         */
        bool allNullOpts();
    };

    const clang::Attr* attr;
    const clang::ForStmt& stmt;
    LoopTypes type = {LoopType::Regular};
    Axises axis = {Axis::Auto};

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};
    std::string tileSize = "";

    AttributedTypeInfo sharedInfo;
    AttributedTypeInfo exclusiveInfo;

    std::optional<OptSizes> overridenInnerSizes;
    std::optional<int> simdLength;

    struct {
        std::string typeName;           ///< Name of type of loop variable.
        std::string name;               ///< Name of loop variable.
        const clang::VarDecl* varDecl;  ///< Declaration node of loop variable.
    } var;
    struct {
        const clang::Expr* start;  ///< Expression node -- start of the loop range.
        const clang::Expr* end;    ///< Expression node -- end of the loop range.
        size_t size = 0;           ///< Size of the loop range. 0 if unknown
    } range;
    struct {
        const clang::BinaryOperator* cmp;  ///< Comparison node.
        BinOp op = BinOp::Eq;              ///< Comparison operator.
    } condition;
    struct {
        const clang::Expr* val;  ///< Expression node -- increment value.
        union {
            UnOp uo;
            BinOp bo;
        } op;  ///< Increment operator.
    } inc;

    /**
     * @brief Checks if a syncronization after this loop should be performed.
     * @return Boolean indicating whether a syncronization should be performed.
     */
    [[nodiscard]] bool shouldSync();
    /**
     * @brief Marks that @shared variable is used in this loop or child loops.
     */
    void markSharedUsed();
    /**
     * @brief   Marks that @exclusive variable is used in this loop or child loops.
     */
    void markExclusiveUsed();

    /**
     * @brief Checks if the loop iteration is incremental
     * @return Boolean indicating whether the loop iteration is incremental
     */
    [[nodiscard]] bool IsInc() const;
    /**
     * @brief Checks if the increment operation is unary
     * @return Boolean indicating whether the increment operation is unary.
     */
    [[nodiscard]] bool isUnary() const;

    /**
     * @brief Retrieves first parent loop that has an attribute.
     * @return Pointer to the parent loop that has an attribute.
     */
    OklLoopInfo* getAttributedParent();

    /**
     * @brief Retrieves first parent loop that satisfies a given condition.
     * @param f The predicate.
     * @return Pointer to the parent loop that satisfies the condition.
     */
    OklLoopInfo* getAttributedParent(std::function<bool(OklLoopInfo&)> f);

    /**
     * @brief Retrieves the first child loop that has an attribute.
     * @return Pointer to the first child loop that has an attribute.
     */
    OklLoopInfo* getFirstAttributedChild();
    /**
     * @brief Retrieves the first child loop that satisfies a given condition.
     * @param f The predicate.
     * @return Pointer to the first child loop that satisfies the condition.
     */
    OklLoopInfo* getFirstAttributedChild(std::function<bool(OklLoopInfo&)> f);

    /**
     * @brief Retrieves the distance to the for loop tree leaf.
     */
    size_t getHeight();

    /**
     * @brief Retrieves the distance to the for loop tree leaf, ignoring loops of other types.
     * @param type The loop type.
     */
    size_t getHeightSameType(const LoopType&);

    /**
     * @brief Retrieves the sizes of all @inner loops of this loop and its children.
     * @return The inner sizes.
     */
    OptSizes getInnerSizes();

    /**
     * @brief Checks if the loop is of a given type.
     * @param type The loop type.
     * @return Boolean indicating whether the loop is of the given type.
     */
    [[nodiscard]] bool is(const LoopType&) const;

    /**
     * @brief Checks if the loop is @tile and has given types.
     * @param type1 The first loop type.
     * @param type2 The second loop type.
     * @return Boolean indicating whether the loop is of the two given types.
     */
    [[nodiscard]] bool is(const LoopType&, const LoopType&) const;

    /**
     * @brief Checks if the loop has a given type.
     * @param type The loop type.
     * @return Boolean indicating whether the loop has the given type.
     */
    [[nodiscard]] bool has(const LoopType&) const;
    /**
     * @brief Checks if the loop is tiled (@tile loop).
     * @return Boolean indicating whether the loop is tiled.
     */
    [[nodiscard]] bool isTiled() const;

    /**
     * @brief Checks if the loop is completely regular.
     * @return Boolean indicating whether the loop is regular.
     */
    [[nodiscard]] bool isRegular() const;

    /**
     * @brief Checks if the loop is of a given axis.
     * @param axis The axis.
     * @return Boolean indicating whether the loop is of the given axis.
     */
    [[nodiscard]] bool is(const Axis&) const;
    /**
     * @brief Checks if the loop is @tile and has given axes
     * @param axis1 The first axis.
     * @param axis2 The second axis.
     * @return Boolean indicating whether the loop is of the two given axes.
     */
    [[nodiscard]] bool is(const Axis&, const Axis&) const;
    /**
     * @brief Checks if the loop has a given axis.
     * @param axis The axis.
     * @return Boolean indicating whether the loop has the given axis.
     */
    [[nodiscard]] bool has(const Axis&) const;

    /**
     * @brief Updates the auto axis with a specific axis if possible.
     * @return Boolean indicating whether the update was successful.
     */
    [[nodiscard]] bool updateAutoWithSpecificAxis();
    /**
     * @brief Checks if this is the last outer loop. Used for semantic check of @shared and
     * @exclusive attributes
     * @return Boolean indicating whether this is the last outer loop.
     */
    [[nodiscard]] bool isLastOuter();
};

/**
 * @struct OklKernelInfo
 * @brief This structure represents information about an OKL kernel.
 *
 * It contains information about the kernel's declaration and top-level outer loops.
 */
struct OklKernelInfo {
    explicit OklKernelInfo(const clang::FunctionDecl& decl)
        : decl(std::ref(decl)){};
    const std::reference_wrapper<const clang::FunctionDecl> decl;  ///< The kernel declaration.
    std::list<OklLoopInfo*> topLevelOuterLoops = {};               ///< The top-level outer loops.
    std::list<OklLoopInfo> topLevelLoops = {};                     ///< The top-level loops.
};
}  // namespace oklt
