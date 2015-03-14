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
      
      af::array forward_prop();
      void backward_prop();
      void allocate_memory();
     
      bool validate();
      /*
       * These function add layers on to the end of the network
       */
      void add_layer(std::unique_ptr<Layer> l);

      template<class C, typename... Params>
      C& construct_layer(Params... params) {
	af::dim4& in_dim=layers.back()->out_dim;
	layers.push_back(std::unique_ptr<C>(new C(in_dim, params...)));
	return *layers.back();
      }

      LossLayer& loss_layer() {
	return *static_cast<LossLayer*>(&(*layers.back()));
      }
     
    protected:
      Batch batch;
      af::array dropout_mask;
    };
  }
}

#endif /* FEED_FORWARD_H */
