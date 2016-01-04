#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include "Layer.hpp"
#include "Batch.hpp"
#include "Optimizable.hpp"
#include "Initialization.hpp"

namespace DeepFire {
  class NeuralNetwork : public Optim::GradientOptimizable {
    
  public:
    /*
     * This must be called before loss or grad if the internal structure is modified.
     */
    virtual void allocate_memory() {};

    /*
     * Performs any assertions we can think of about the internal structure of the network.
     */
    virtual bool validate()=0;
    
    virtual af::array& weights() { return w; }

    virtual void use_batch(Batch b) {
      batch=b;
    };

    virtual af::array loss()=0;
    virtual const af::array& grad()=0;
    
    std::vector<std::unique_ptr<Layer>> layers;
    
  protected:
    af::array w;
    /*
     * The batch is stored during computation to calculate loss, etc, but you 
     * should never ask a Neural Network what batch it's working on.  That's
     * just... bad manners.
     */
    Batch batch;
  };
}

#endif /* NEURAL_NETWORK_H */
