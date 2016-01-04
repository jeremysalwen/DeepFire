#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <memory>
#include <arrayfire.h>
#include "Layer.hpp"
#include "Batch.hpp"
#include "LossLayer.hpp"
#include "NeuralNetwork.hpp"
namespace DeepFire {
  namespace NN {
    class FeedForward : public NeuralNetwork {
    public:
      virtual void allocate_memory();
      virtual bool validate();

      virtual void use_batch(Batch b);
      
      virtual af::array loss();
      virtual const af::array& grad();
           
      /*
       * These function add layers on to the end of the network
       */
      void add_layer(std::unique_ptr<Layer> l);

      template<class C, typename... Params>
      C& construct_layer(Params... params) {
	af::dim4& in_dim=layers.back()->out_dim;
	C* layer=new C(in_dim, params...);
	layers.push_back(std::unique_ptr<Layer>(layer));
	return *layer;
      }

      LossLayer& loss_layer() {
	return *static_cast<LossLayer*>(&(*layers.back()));
      }

    protected:
      af::array g;
     
     };
  }
}

#endif /* FEED_FORWARD_H */
