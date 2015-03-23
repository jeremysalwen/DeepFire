#include "MomentumOptimizer.hpp"

#include <arrayfire.h>
namespace DeepFire {
  namespace Optim {

    void MomentumOptimizer::reset_training() {
      m=af::constant(0.0,optim->weights().dims(),af::dtype::f64);
    }

    void MomentumOptimizer::set_optimizable(GradientOptimizable& o) {
      optim=&o;
      reset_training();
    }

    int MomentumOptimizer::optimize(int iterations) {
      af::array& weights=optim->weights();
      for (int i=0; i < iterations; i++) {
	Batch b=dataset->sample_random(batch_size);
	optim->use_batch(b);
	
	{
	  af::array loss=optim->loss();
	}
	af::array grad=optim->grad();

	m=m*momentum+grad*learning_rate;
	weights-=m;
      }

      return iterations;
    }
  }
}
