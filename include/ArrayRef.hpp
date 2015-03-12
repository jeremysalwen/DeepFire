#ifndef ARRAY_REF_H
#define ARRAY_REF_H

#include <arrayfire.h>
#include <assert.h>
namespace DeepFire {
    class ArrayDeref {
    public:
      ArrayDeref(af::array& arr, af::seq ind, af::dim4 dims): arr(arr), ind(ind), dims(dims) {}
      template <typename T>
      operator T() {
	return af::moddims(arr(ind),dims);
      }
      
      template<typename T>
      ArrayDeref& operator=(T arg) {
	af::moddims(arr(ind),dims)=arg;
	return *this;
      }
      
    protected:
      af::array arr;
      af::seq ind;
      af::dim4 dims;
    };
  
  class ArrayRef {
  public:
    ArrayRef(): arr(NULL) {}
    ArrayRef(af::array& array, af::seq ind, af::dim4 dims):arr(&array), ind(ind), dims(dims) {
      assert(ind.s.step==1);
    }
    ArrayRef(af::array& array) :arr(&array),ind(af::span),dims(array.dims()) {}

    ArrayDeref operator*() const {
      return ArrayDeref(*arr, ind, dims);
    }
    
    ArrayRef operator()(af::seq i) const {
      assert(i.s.step==1);
      assert(i.s.end-i.s.begin<=ind.s.end-ind.s.begin);
      assert(dims[1]==1 &&  dims[2]==1 && dims[3]==1);
      af::seq nseq=af::seq(ind.s.begin+i.s.begin,ind.s.begin+i.s.end-ind.s.begin);
      af::dim4 ndims=af::dim4(i.s.end=i.s.begin+1);
      return ArrayRef(*arr, nseq, ndims);
    }
  protected:
    af::array* arr;
    af::seq ind;
    af::dim4 dims;
  };
}

#endif /* ARRAY_REF_H */
