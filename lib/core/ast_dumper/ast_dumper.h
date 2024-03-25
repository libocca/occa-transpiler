#pragma once

#include <llvm/Support/raw_ostream.h>

namespace clang {
class Decl;
class Stmt;
class QualType;
class Type;
class APValue;
}  // namespace clang

namespace oklt {
class SessionStage;

LLVM_DUMP_METHOD void dump(clang::Decl* D,
                           SessionStage& stage,
                           llvm::raw_ostream& os = llvm::errs());
LLVM_DUMP_METHOD void dump(clang::Stmt* S,
                           SessionStage& stage,
                           llvm::raw_ostream& os = llvm::errs());
LLVM_DUMP_METHOD void dump(clang::QualType* qt,
                           SessionStage& stage,
                           llvm::raw_ostream& os = llvm::errs());
LLVM_DUMP_METHOD void dump(clang::Type* t,
                           SessionStage& stage,
                           llvm::raw_ostream& os = llvm::errs());
LLVM_DUMP_METHOD void dump(clang::APValue* ap,
                           SessionStage& stage,
                           llvm::raw_ostream& os = llvm::errs());

}  // namespace oklt
