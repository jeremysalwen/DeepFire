#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include "Layer.hpp"
#include "Batch.hpp"
#include "GradientOptimizable.hpp"
#include "Initialization.hpp"

namespace DeepFire {
  class NeuralNetwork : Optim::GradientOptimizable {
    friend class Initializer<NeuralNetwork>;
    
  public:
    virtual bool validate()=0;
    
    virtual af::array& weights() { return weights; }

    virtual void use_batch(Batch b) {
      batch=b;
    };

    virtual af::array loss()=0;
    virtual af::array& grad()=0;
    
    std::vector<std::unique_ptr<Layer>> layers;
    
  protected:
    af::array weights;
    /*
     * The batch is stored during computation to calculate loss, etc, but you 
     * should never ask a Neural Network what batch it's working on.  That's
     * just... bad manners.
     */
    Batch batch;
  };
}

#endif /* NEURAL_NETWORK_H */
