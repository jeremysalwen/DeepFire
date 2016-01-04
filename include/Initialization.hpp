#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <memory>

#include <arrayfire.h>
#include "ArrayRef.hpp"

namespace DeepFire {

  template <class C>
  class Initializer  {
  public:
    virtual void initialize(C* object)=0;
  };

}


#endif /* INITIALIZATION_H */
