#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
#include <arrayfire.h>
#include "Layer.hpp"
#include "CombinedLayer.hpp"
#include "Util.hpp"

namespace DeepFire {
  namespace Activation {
    
    class Sigmoid : public Layer {
    public:
      Sigmoid(const af::dim4& dim) : Layer(dim) {}
      virtual inline  af::array forward_prop(const af::array& in) {
	last_output=1/(1+af::exp(in));
	return last_output;
      }
      virtual inline af::array backward_prop(const af::array& gradin) {
	return gradin*last_output*(1-last_output);
      }
    protected:
          af::array last_output;
    };
    
    class Tanh : Layer {
    public:
      Tanh(const af::dim4& dim) : Layer(dim) {}
      virtual inline af::array forward_prop(const af::array& in) {
	last_output=af::tanh(in);
	return last_output;
      }
      virtual inline af::array backward_prop(const af::array& gradin) {
	return gradin*(1-last_output*last_output);
      }
    protected:
      af::array last_output;
    };

    class ReLU : Layer {
    public:
      ReLU(const af::dim4& dim): Layer(dim) {}
      virtual inline af::array forward_prop(const af::array& in) {
	last_input=in;
	return af::max(in, 0);
      }
      virtual inline af::array backward_prop(const af::array& gradin) {
	return gradin*(last_input > 0);
      }
    protected:
      af::array last_input;
    };

    class SoftPlus : Layer {
    public:
      SoftPlus(const af::dim4& dim): Layer(dim) {}
      virtual inline af::array forward_prop(const af::array& in) {
	last_input=in;
	return af::log(1+af::exp(in));
      }

      
      virtual inline af::array backward_prop(const af::array& gradin) {
	return gradin/(1+af::exp(last_input));
      }
    protected:
            af::array last_input;
    };
    
    class Exp : Layer {
     public:
      Exp(const af::dim4& dim) : Layer(dim) {
      }
      virtual af::array forward_prop(const af::array& in) {
	last_output=af::exp(in);
	return last_output;
      }
      virtual af::array backward_prop(const af::array& gradin) {
	return gradin*last_output;
      }
    protected:
       af::array last_output;
    };

    class Normalize : Layer {
    public:
      Normalize(const af::dim4& dim) : Layer(dim) {
	nelems=nelements_batch(dim);
      }
      virtual inline af::array forward_prop(const af::array& in) {
	af::array flattened=flat_batch(in); //turn every sample into a flat array
	sum=af::sum(flattened,1);
	last_output=moddim_batch(flattened/tile(sum,nelems),out_dim);
	return last_output;
      }
      virtual inline af::array backward_prop(const af::array& gradin) {
	af::array flattened=flat_batch(gradin);
	af::array result=gradin*(1-last_output)/tile(sum,1,nelems);
	return moddim_batch(result,out_dim);
      }
      
    protected:
      dim_type nelems;
      af::array sum;
      af::array last_output;
    };

    /*
     * Softmax layer with Log valued outputs
     */
    class SoftMax : Layer {
    public:
      SoftMax(const af::dim4& dim): Layer(dim){
	nelems=nelements_batch(dim);
      };
      virtual inline af::array forward_prop(const af::array& in) {
	af::array flattened=moddim_batch(in,af::dim4(1,nelems));
	
	//Flattened computation
	last_exp=af::exp(flattened);
	sum=af::sum(last_exp,1);
	af::array result=flattened-af::tile(af::log(sum),1,nelems);
	
	return moddim_batch(result,out_dim);
      }
      
      virtual inline af::array backward_prop(const af::array& gradin) {
	af::array flattened=moddim_batch(gradin,af::dim4(1,nelems));

	af::array result=flattened*(1-last_exp/af::tile(sum,1,nelems));
	return moddim_batch(result,in_dim);
      }
      dim_type nelems;
      af::array sum;
      af::array last_exp;
    };
  }
}

#endif
