#ifndef SPARSE_INITIALIZATION_H
#define SPARSE_INITIALIZATION_H

#include "ConnectionLayer.hpp"
#include "Initialization.hpp"

namespace DeepFire {
  class SparseInitializer : public Initializer<Connection::ConnectionLayer> {
  public:
    SparseInitializer(double fanin, double magnitude=1, bool normal=true, double mean=0):fanin(fanin), magnitude(magnitude), normal(normal), mean(mean) {
    }

    virtual void initialize(Connection::ConnectionLayer* C);
    
    double fanin;
    double magnitude;
    bool normal;
    double mean;
  };
}

#endif /* SPARSE_INITIALIZATION_H */
