#ifndef GRADIENT_OPTIMIZABLE_H
#define GRADIENT_OPTIMIZABLE_H

#include <arrayfire.h>
#include "Batch.hpp"

namespace DeepFire {
  namespace Optim {
    class Optimizable {
      /*
       * Returns a reference to the weights that are to be optimized.
       */
      virtual af::array& weights()=0;

      /*
       * Set the current batch of samples we are cacluating with
       */
      virtual void use_batch(Batch b)=0;

      /* 
       * Evaluates the loss using the last Batch passed to use_batch.
       */
      virtual af::array loss()=0;
    };
    
    class GradientOptimizable : public Optimizable {

      /*
       * Evaluates the gradient of the loss  WRT the weights using 
       * the last Batch passed to use_batch.
       *
       * Note that for memory reasons grad() does not return the gradient per-sample
       * (as would be most logically consistent, but just the sum.
       */
      virtual const af::array& grad()=0;
    };
    
    /* 
     * Hessian-Vector-Product optimizable.
     * This can also use an approximation of the Hessian instead of the actual Hessian matrix.
     */
    class HVPOptimizable : public GradientOptimizable {
      virtual af::array& HVP(const af::array vec)=0;
    };
  }
}

#endif /* GRADIENT_OPTIMIZABLE_H */
