#pragma once

#include <clang/AST/AST.h>

namespace oklt {


struct OuterForStmt {
  clang::AttributedStmt *outerForStmt;
  std::vector<clang::AttributedStmt> innerForStmts;
};

}

//typedef float** Matrix @dim(100, 100);

//// FunctionCtx push to Stage
////FunctionDecl
//@kernel // <-- handler ( __global__, + param sycl::ctx, sycl::quueu
//void func1(const float Matrix @restrict, // handle <--- __restrict__ const float *ptr,
//        int size, const double *OridnaryCPPArg) {

//  @outer for(...) { // travesrse => VALIDATE
//    @inner for(...) { // traverse => VALIDATE
//      // traverse => VALIDATE
//      @shared arr[100];
//      // handle => __shared__, Complex Op
//    } //handler
//  } // handle !!!!  // check isFirst(out) && HasFunctionDescription => handler

//  // <--- Check point (
//  @outer for(...) { // traverse => VALIDATE
//    @inner for(...) { //traverse => VALIDATE
//      //traverse
//      @shared arr2[100];
//      //handle
//    } //handle
//  } //handle  // check isFirst(out) && HasFunctionDescription => handler

//  // <--- Check point
//  @outer for(...) {
//    @inner for(...) {
//      @shared arr[100];
//    }
//  }

//  // <--- Check point
//  @outer for(...) {
//    @inner for(...) {
//      @shared arr[100];
//    }
//  }
//}

////Conclusion: complexity - N + (N-1) traverses

//__global__ void func1_1(const float *ptr, // handle <--- __restrict__ const float *ptr,
//                        int size)
//{
//  //outer for 2
//  @shared arr2[100];
//}
