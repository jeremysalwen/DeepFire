#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <arrayfire>

namespace DeepFire {

  /*
   * Note that this interface can be used to initialize a Layer, or a whole network
   */
  template <class C>
  class Initialization {
  public:
    virtual void initialize(C* object)=0;
  }

    class 
}

#endif /* INITIALIZATION_H */
