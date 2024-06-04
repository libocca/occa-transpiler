#pragma once

//INFO: from documentation 
//  isNaN
//  Description:The function returns 1, if and only if its argument is a NaN.
//  Calling interface:
//    int __binary32_isNaN(float x);

bool okl_is_nan(float value) {
  return __binary32_isNaN(value) == 1;
}
