#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CONST_QUALIFIER = "const";
const std::string HIP_CONST_QUALIFIER = "__constant__";

bool isConstantSizeArray(const VarDecl* var) {
    return var->getType().getTypePtr()->isConstantArrayType();
}

bool isPointer(const VarDecl* var) {
    return var->getType()->isPointerType();
}

bool isPointerToConst(const VarDecl* var) {
    return isPointer(var) && var->getType()->getPointeeType().isLocalConstQualified();
}

bool isConstPointer(const VarDecl* var) {
    return isPointer(var) && var->getType().isLocalConstQualified();
}

bool isConstPointerToConst(const VarDecl* var) {
    return isPointerToConst(var) && isConstPointer(var);
}

bool isGlobalConstVariable(const clang::Decl* decl) {
    // Should be variable declaration
    if (!isa<VarDecl>(decl)) {
        return false;
    }

    auto var = dyn_cast<VarDecl>(decl);

    // Should be global variable
    if (var->isLocalVarDecl() && !var->hasGlobalStorage()) {
        return false;
    }

    // pointer to const
    if (isPointer(var)) {
        return isPointerToConst(var);
    }

    auto type = var->getType();
    // Should be constant qualified
    if (!(type.isLocalConstQualified() || type.isConstant(var->getASTContext()))) {
        llvm::outs() << var->getDeclName().getAsString() << " is not constant\n";
        return false;
    }

    return true;
}

std::string getNewDeclStrConstantArray(const VarDecl* var) {
    auto* arrDecl = dyn_cast<ConstantArrayType>(var->getType().getTypePtr());

    auto unqualifiedTypeStr = arrDecl->getElementType().getLocalUnqualifiedType().getAsString();

    auto type = arrDecl->getElementType();
    type.removeLocalConst();
    auto qualifiers = type.getQualifiers();

    auto varName = var->getDeclName().getAsString();  // Name of variable

    std::string newDeclStr;
    if (qualifiers.hasQualifiers()) {
        auto noConstQualifiersStr = qualifiers.getAsString();
        newDeclStr = noConstQualifiersStr + " " + HIP_CONST_QUALIFIER + " " + unqualifiedTypeStr +
                     " " + varName;
    } else {
        newDeclStr = HIP_CONST_QUALIFIER + " " + unqualifiedTypeStr + " " + varName;
    }
    return newDeclStr;
}

std::string getNewDeclStrVariable(const VarDecl* var) {
    auto unqualifiedTypeStr = var->getType().getLocalUnqualifiedType().getAsString();

    auto type = var->getType();
    type.removeLocalConst();
    auto qualifiers = type.getQualifiers();

    auto VarName = var->getDeclName().getAsString();  // Name of variable

    std::string newDeclStr;
    if (qualifiers.hasQualifiers()) {
        auto noConstQualifiersStr = qualifiers.getAsString();
        newDeclStr = noConstQualifiersStr + " " + HIP_CONST_QUALIFIER + " " + unqualifiedTypeStr +
                     " " + VarName;
    } else {
        newDeclStr = HIP_CONST_QUALIFIER + " " + unqualifiedTypeStr + " " + VarName;
    }
    return newDeclStr;
}

std::string getNewDeclStrPointerToConst(const VarDecl* var) {
    auto type = var->getType();

    auto unqualifiedPointeeType = type->getPointeeType();
    unqualifiedPointeeType.removeLocalConst();
    auto unqualifiedPointeeTypeStr = unqualifiedPointeeType.getAsString();

    auto varName = var->getDeclName().getAsString();

    std::string newDeclStr;
    if (type.hasQualifiers()) {
        auto qualifiersStr = type.getQualifiers().getAsString();
        newDeclStr = HIP_CONST_QUALIFIER + " " + unqualifiedPointeeTypeStr + " * " + qualifiersStr +
                     " " + varName;
    } else {
        newDeclStr = HIP_CONST_QUALIFIER + " " + unqualifiedPointeeTypeStr + " * " + varName;
    }
    return newDeclStr;
}
}  // namespace

namespace oklt::cuda_subset {
bool handleGlobalConstant(const clang::Decl* decl, SessionStage& s) {
    if (!isGlobalConstVariable(decl)) {
        return true;
    }

    auto var = dyn_cast<VarDecl>(decl);

#ifdef TRANSPILER_DEBUG_LOG
    auto type_str = var->getType().getAsString();
    auto declname = var->getDeclName().getAsString();

    llvm::outs() << "[DEBUG] Found constant global variable declaration:"
                 << " type: " << type_str << ", name: " << declname << "\n";
#endif

    std::string newDeclStr;
    if (isConstantSizeArray(var)) {
        newDeclStr = getNewDeclStrConstantArray(var);
    } else if (isPointerToConst(var)) {
        newDeclStr = getNewDeclStrPointerToConst(var);
        s.pushWarning("__constant__ applied to pointer type");
    } else {
        newDeclStr = getNewDeclStrVariable(var);
    }

    // volatile const int var_const = 0;
    // ^                          ^
    // start_loc                  end_loc
    auto start_loc = var->getBeginLoc();
    auto end_loc = var->getLocation();
    auto range = SourceRange(start_loc, end_loc);

    auto& rewriter = s.getRewriter();
    rewriter.ReplaceText(range, newDeclStr);

    return true;
}
}  // namespace oklt::cuda_subset
