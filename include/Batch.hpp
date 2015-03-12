#ifndef BATCH_H
#define BATCH_H

#include <arrayfire.h>

namespace DeepFire {
  class Batch {
  public:
    af::array data;
    af::array label;
  };
}

#endif /* BATCH_H */
