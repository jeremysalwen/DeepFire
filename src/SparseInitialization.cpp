#include "SparseInitialization.hpp"

namespace DeepFire {
  void SparseInitializer::initialize(Connection::ConnectionLayer* C) {
    //This will be the fraction of weights that need to be active
    double prob=fanin/C->fanin();

    ArrayRef weights = C->weights;
    af::array r;
    if(normal) {
      r=af::randn(C->num_weights);
    } else {
      r=af::randu(C->num_weights);
    }
       
    *weights=r*(af::randn(C->num_weights)<=prob)*magnitude+mean;
  }
}
