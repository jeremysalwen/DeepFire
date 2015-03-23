#ifndef MOMENTUM_OPTIMIZER_H
#define MOMENTUM_OPTIMIZER_H

#include "Optimizer.hpp"

namespace DeepFire {
  namespace Optim {
    
    class MomentumOptimizer : GradientOptimizer {
    public:
      MomentumOptimizer(dim_type batch_size, double learning_rate=0.001, double momentum=0.9, bool nesterov=false):
	batch_size(batch_size),
	learning_rate(learning_rate),
	momentum(momentum),
	nesterov(nesterov),
	optim(NULL),
	dataset(NULL) {
	assert(learning_rate>0);
	assert(momentum>0);
      }
      
      virtual void set_optimizable(GradientOptimizable& o); 

	virtual void set_train_dataset(Dataset& d) {
	  dataset=&d;
	}

	virtual void reset_training();
	virtual int optimize(int iterations);


	dim_type batch_size;
	double learning_rate;
	double momentum;
	bool nesterov;
      protected:
	GradientOptimizable* optim;
	Dataset* dataset;
	af::array m;

      };
      
    }
  }


#endif /* MOMENTUM_OPTIMIZER_H */
