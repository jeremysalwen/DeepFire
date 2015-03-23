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
      virtual void set_optimizable(HVPOptimizable& opt)=0;

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
      virtual void set_optimizable(GradientOptimizable& opt);
      virtual void set_optimizable(HVPOptimizable& opt) {
	set_optimizable(static_cast<GradientOptimizable&>(opt));
      }
    };

    class Optimizer : GradientOptimizer {
   
      virtual void set_optimizable(Optimizable& opt) =0;
      virtual void set_optimizable(GradientOptimizable& opt) {
	set_optimizable(static_cast<Optimizable&>(opt));
      }
    };
  }
}

#endif /* OPTIMIZER_H */
