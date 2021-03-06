#ifndef CONNECTION_LAYER_H
#define CONNECTION_LAYER_H
#include <arrayfire.h>
#include "Layer.hpp"
#include "Util.hpp"

namespace DeepFire {
  namespace Connection {

    class ConnectionLayer : public Layer {
    public:
      using Layer::Layer;

      /*
       * Note that the fanin must be constant for all outputs
       */
      virtual int fanin()=0;

    };
    
    /*
     * Note that since we are using the first dimension for the sample size, 
     * the samples are row vectors.  We use Transpose matrix multiplication to operate
     * on them as row vectors
     */
    class FullyConnected : public ConnectionLayer {
    public:
      FullyConnected(const af::dim4& input_dim, const af::dim4& output_dim):
	ConnectionLayer(in_dim,out_dim,(nelements_batch(in_dim)+1)*nelements_batch(out_dim)),
	in_dim_flat(1,nelements_batch(in_dim)),
	out_dim_flat(1,nelements_batch(out_dim))
      {
	num_weights=(in_dim_flat[1]+1)*out_dim_flat[1];
      }

      virtual inline af::array forward_prop(const af::array& in) {
	last_input=moddim_batch(in, in_dim_flat);
	
	//Flattened operation
	af::array last_output=af::matmulNT(*weights,last_input);
	
	return moddim_batch(last_output,out_dim_flat);
      }
      virtual inline af::array backward_prop(const af::array& in) {
	af::array in_flat=moddim_batch(in,out_dim_flat);

	//Flattened operation
	*grad=af::flat(af::matmulNT(in_flat,last_input));
	af::array out_flat=af::matmulTT(*weights,in_flat);
	
	return moddim_batch(out_flat,out_dim_flat);
      }

      virtual int fanin() {return in_dim_flat[1];}

    protected:
      af::dim4 in_dim_flat, out_dim_flat;
      af::array last_input;
    };

  }
}
#endif
