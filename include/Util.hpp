#ifndef UTIL_H
#define UTIL_H

#include <arrayfire.h>
#include <memory>

/*
 *  These functions operate on arrays of samples /ignoring/ the first dimension.
 */

inline dim_type nelements_batch(const af::dim4& dim) {
  return dim[1]*dim[2]*dim[3];
}

inline af::array flat_batch(const af::array& A) {
  dim_type data_size=nelements_batch(A.dims());
  return af::moddims(A, af::dim4(A.dims(0), data_size));
}

inline af::array moddim_batch(const af::array& A, af::dim4 dim) {
  dim[0]=A.dims(0);
  return af::moddims(A, dim);
}

inline bool dim_eq_batch(const af::dim4& dim1, const af::dim4& dim2) {
  return dim1[1]==dim2[1] && dim1[2]==dim2[2] && dim1[3]==dim2[3];
}


#endif /* UTIL_H */
