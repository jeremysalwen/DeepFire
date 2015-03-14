#ifndef LAYER_H
#define LAYER_H

#include <arrayfire.h>
#include "ArrayRef.hpp"
#include "Initialization.hpp"

namespace DeepFire {

  /*
   * Note that the first dimension is ignored, as it is used for the sample number
   *
   */
  class Layer {
    friend class Initialization<Layer>;
  public:
    af::dim4 in_dim;
    af::dim4 out_dim;

    int num_weights;

    Layer(const af::dim4& dim) :in_dim(dim), out_dim(dim),num_weights(0) {}
    Layer(const af::dim4 in_dim, const af::dim4& out_dim, int num_weights=0) :in_dim(out_dim), out_dim(out_dim),num_weights(num_weights) {}

    /*
     * These functions give the Layer an ArrayRef where it should
     * store its optimizable weights.  It should store num_weights elements.
     */
    
    virtual void use_dropout_mask(ArrayRef ref) {}
    virtual void use_weights(ArrayRef weights, ArrayRef grad) {}
    
    /*
     * Note that backward_prop should be called after forward_prop, and it 
     * assumes the input values are the same/
     *
     * For backprop, the first dim of in should be 1, as the derivative
     * has already been summed over all samples.
     */
    virtual af::array forward_prop(const af::array& in)=0;
    virtual af::array backward_prop(const af::array& in)=0;

    //By default a layer should not
    virtual bool loss_layer() { return false; }
    virtual void use_labels(const af::array& labels) {}
  };

  class FlatLayer : Layer {
    
  };
}
#endif
