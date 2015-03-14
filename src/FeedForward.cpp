#include "FeedForward.hpp"
#include "Util.hpp"

namespace DeepFire {
  namespace NN {
    bool FeedForward::validate() {
      if(layers.size()<1) {
	return false;
      }
      if(!layers.back()->loss_layer()) {
	return false;
      }
      for(unsigned int i=0; i<layers.size(); i++) {
	if(layers[i]->loss_layer()) {
	  return false;
	}
      }
      for(unsigned int i=1; i<layers.size(); i++) {
	if(!dim_eq_batch(layers[i-1]->out_dim,layers[i]->in_dim)) {
	  return false;
	}
      }
      return true;
    }
    
    void FeedForward::add_layer(std::unique_ptr<Layer> l) {
      layers.push_back(std::move(l));
    }
    
    void FeedForward::allocate_memory() {
      dim_type num_weights=0;
      for(auto& L :layers)  {
	num_weights+=L->num_weights;
      }
      w=af::array(num_weights);
      g=af::array(num_weights);

      dim_type i=0;
      for(auto& L: layers) {
	dim_type num_weights=L->num_weights;
	af::seq ind(i,i+num_weights-1);
	af::dim4 dim(1,num_weights);
	ArrayRef wref(w,ind,dim);
	ArrayRef gref(g,ind,dim);
	L->use_weights(wref,gref);
      }
    }

    void FeedForward::use_batch(Batch b) {
      batch=b;
      layers.back()->use_labels(b.label);  
    }
    af::array FeedForward::loss() {
      af::array data=batch.data;
      for( auto& L : layers) {
	data=L->forward_prop(data);
      }
      return data;
    }
    const af::array&   FeedForward::grad() {
      af::array grad= loss_layer().backward_prop();
      for(int i=layers.size()-1; i>=0; i--) {
	grad=layers[i]->backward_prop(grad);
      }
      return g;
    }
  }
}
