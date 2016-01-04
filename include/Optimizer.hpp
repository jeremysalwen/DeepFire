#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <assert.h>
#include <arrayfire.h>
#include "Optimizable.hpp"
#include "Dataset.hpp"

namespace DeepFire {
  namespace Optim {
    class HVPOptimizer{
    public:
      virtual void set_optimizableH(HVPOptimizable& opt)=0;

      virtual void set_train_dataset(Dataset& d)=0;

      virtual void set_devel_dataset(Dataset& d) {};

      /*
       * This only resets internal variables, but keeps the dataset an optimizable the same
       */
      virtual void reset_training()=0;
      
      /*
       * Reurns the number of iterations performed (if it's less than input, it converged)
       */
      virtual int optimize(int iterations)=0;
    };

    class GradientOptimizer : HVPOptimizer{
    public:  
      virtual void set_optimizableH(HVPOptimizable& opt) {
	set_optimizableG(static_cast<GradientOptimizable&>(opt));
      }
      virtual void set_optimizableG(GradientOptimizable& opt)=0;
    };

    class Optimizer : GradientOptimizer {
    public:
      virtual void set_optimizableG(GradientOptimizable& opt) {
	set_optimizableV(static_cast<Optimizable&>(opt));
      }

      virtual void set_optimizableV(Optimizable& opt) =0;

    };
  }
}

#endif /* OPTIMIZER_H */
