#ifndef COMBINED_LAYER_H
#define COMBINED_LAYER_H
#include <arrayfire.h>
#include "Layer.hpp"

namespace DeepFire {
  template <class L1, class L2>
  class CombinedLayer :Layer {
    L1 l1;
    L2 l2;
  public:
    CombinedLayer(const af::dim4& dim): Layer(dim), l1(dim), l2(dim) {
      set_num_weights();
    }
    CombinedLayer(const af::dim4& in_dim, const af::dim4& out_dim): Layer(in_dim, out_dim), l1(in_dim), l2(out_dim) {
      set_num_weights();
    }
    CombinedLayer(const af::dim4& in_dim, const af::dim4& out_dim, const af::dim4& m_dim): Layer(in_dim, out_dim), l1(in_dim,m_dim), l2(m_dim,out_dim) {
      set_num_weights();
    }
    CombinedLayer(L1 l1, L2 l2): Layer(l1.in_dim, l2.out_dim), l1(l1), l2(l2) {
      set_num_weights();
    }
    virtual inline af::array forward_prop(const af::array& in) {
      return l2.forward_prop(l1.forward_prop(in));
    }
    virtual inline af::array backward_prop(const af::array& in) {
      return l1.backward_prop(l2.backward_prop(in));
    }
    virtual inline void use_weights(ArrayRef weights, ArrayRef grad) {
      ArrayRef l1weights=weights(af::seq(0,l1.num_weights-1));
      ArrayRef l2weights=weights(af::seq(0,l2.num_weights-1));

      ArrayRef l1grad=grad(af::seq(0,l1.num_weights-1));
      ArrayRef l2grad=grad(af::seq(0,l2.num_weights-1));

      l1.use_weights(l1weights,l1grad);
      l2.use_weights(l2weights,l2grad);
    }
  protected:
    void set_num_weights() {
      num_weights=l1.num_weights+l2.num_weights;
    }
  };
}
#endif
