#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "Optimizer.hpp"

namespace DeepFire {
  namespace Optim {
    
    class SGDOptimizer : Optimizer {
    public:
      SGDOptimizer(dim_type batch_size, double learning_rate=0.001, double decay=1) :
	batch_size(batch_size),
	learning_rate(learning_rate),
	decay(decay),
	optim(NULL),
	dataset(NULL) {
	
	assert(learning_rate>0);
	assert(decay>0);
      }
      
      virtual void set_optimizable(GradientOptimizable& o) {
	optim=&o;
      }
      
      virtual void set_train_dataset(Dataset& d) {
	dataset=&d;
      }

      virtual void reset_training();
      virtual int optimize(int iterations);


      dim_type batch_size;
      double learning_rate;
      double decay;
      
    protected:
      GradientOptimizable* optim;
      Dataset* dataset;
      double cur_rate;

    };

  }
}

#endif /* SGD_OPTIMIZER_H */
