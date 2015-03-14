#ifndef DATASET_H
#define DATASET_H

#include <arrayfire.h>

#include "Batch.hpp"
namespace DeepFire {
  /*
   * It's up to the implementation of the Dataset to decide what order to provide the samples.
   * This may change in the future.
   */
  class Dataset {
  public:
    virtual dim_type size()=0;
    virtual Batch sample(dim_type num_samples)=0;
  };
}

#endif /* DATASET_H */
