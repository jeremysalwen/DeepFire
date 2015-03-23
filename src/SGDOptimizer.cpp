#include "SGDOptimizer.hpp"

namespace DeepFire {
  namespace Optim {
    void SGDOptimizer::reset_training() {
      cur_rate=learning_rate;
    }
    
    int SGDOptimizer::optimize(int iterations) {
      af::array& weights=optim->weights();
      for (int i=0; i < iterations; i++) {
	Batch b=dataset->sample_random(batch_size);
	optim->use_batch(b);

	{
	  af::array loss=optim->loss();
	}
	af::array grad=optim->grad();
	  
	weights-=grad*cur_rate;
	cur_rate*=decay;
      }
      
      return iterations;
    }

  }
}
