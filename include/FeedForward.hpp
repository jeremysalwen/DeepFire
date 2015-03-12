#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <memory>
#include <arrayfire.h>
#include "Layer.hpp"
#include "Batch.hpp"
#include "LossLayer.hpp"

namespace DeepFire {
  namespace NN {
    class FeedForward {
    public:
      dim_type num_weights;

      void use_batch(Batch b);
      
      af::array forward_prop();
      void backward_prop();
      void allocate_memory();
     
      bool validate();
      
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
      af::array weights;
      std::vector<std::unique_ptr<Layer>> layers;
    };
  }
}

#endif /* FEED_FORWARD_H */
