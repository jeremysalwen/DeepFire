#ifndef DATASET_H
#define DATASET_H

#include <arrayfire.h>

#include "Batch.hpp"
namespace DeepFire {

  class Dataset {
  public:
    virtual dim_type size()=0;

    virtual void shuffle()=0;

    virtual Batch sample_random(dim_type num_samples)=0;
    virtual Batch sample_sequential(dim_type num_samples)=0;

    virtual void reset_sequence()=0;
  };
}

#endif /* DATASET_H */
