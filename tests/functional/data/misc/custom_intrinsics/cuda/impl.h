#pragma once

//INFO: transpiling to cuda backend already include necessary header

bool okl_is_nan(float value) {
  return isnan(value) != 0;
}
