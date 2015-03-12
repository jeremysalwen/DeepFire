#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include <arrayfire.h>
#include "Layer.hpp"
#include "Util.hpp"

namespace DeepFire {
  class LossLayer : public Layer {
  public:
    LossLayer(const af::dim4& dim): Layer(dim, af::dim4(1)) {}
    //Output scalar type
    virtual af::array forward_prop(const af::array& in)=0;

    virtual inline af::array backward_prop(const af::array& in) {
      return backward_prop();
    }
      
    virtual af::array backward_prop()=0;

    virtual bool loss_layer() { return true; }
    virtual void use_labels(const af::array& labels) =0;
  };

  namespace Loss {    
    class MSE : public LossLayer {
    public:
      MSE(const af::dim4& dims) : LossLayer(dims) {}
      virtual inline af::array forward_prop(const af::array& in) {
	af::array flattened=flat_batch(in);
	last_err=flattened-labels;
	return af::sum(af::pow(last_err, 2),1);	
      }

      virtual inline af::array backward_prop() {
	return moddim_batch(2*last_err,in_dim);
      }
      virtual inline void use_labels(af::array& l) {
	labels=flat_batch(l);
      }
    protected:
      af::array labels;
      af::array last_err;
    };
    
    //This assumes inputs are log probabilities
    class CrossEntropy : public LossLayer {
    public:
      CrossEntropy(const af::dim4& dim) : LossLayer(dim) {
	num_classes=nelements_batch(dim);
      }
      virtual inline af::array forward_prop(const af::array& in) {
	af::array flattened=flat_batch(in).T();
	num_samples=flattened.dims(1);
	return -flattened(labels).T();
      }
      virtual inline af::array backward_prop() {
	af::array arr=af::constant(0, num_classes, num_samples);
	arr(labels)=-1;
	return arr;
      }
      virtual inline void use_labels(af::array& l) {
	labels=flat_batch(l);
      }
    protected:
      dim_type num_samples;
      dim_type num_classes;
      af::array labels;
    };
  }
}

#endif /* LOSS_LAYER_H */
