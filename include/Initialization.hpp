#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <arrayfire.h>

namespace DeepFire {

  /*
   * Note that this interface can be used to initialize a Layer, or a whole network
   */
  template <class C>
  class Initialization {
  public:
    virtual void initialize(C* object)=0;
  };

}

#endif /* INITIALIZATION_H */
