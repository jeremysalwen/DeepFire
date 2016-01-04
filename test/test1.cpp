#include "ActivationFunction.hpp"
#include "ConnectionLayer.hpp"
#include "Optimizable.hpp"
#include "Dataset.hpp"
#include "FeedForward.hpp"
#include "Initialization.hpp"
#include "SparseInitialization.hpp"
#include "NeuralNetwork.hpp"
#include "FeedForward.hpp"
#include "ConnectionLayer.hpp"
#include "SGDOptimizer.hpp"

using namespace DeepFire;
int main(int argc, char** argv) {
  NN::FeedForward network;

  std::shared_ptr<Connection::FullyConnected> layer(new Connection::FullyConnected(af::dim4(1,784), af::dim4(1,10)));

  network.construct_layer<Connection::FullyConnected>(af::dim4(1,784));
  network.construct_layer<Activation::Sigmoid>();
  network.construct_layer<Connection::FullyConnected>(af::dim4(1,10));
  network.construct_layer<Loss::CrossEntropy>();

  network.validate();
  network.allocate_memory();

  Dataset data=new Dataset();
  Optim::SGDOptimizer optimizer(10);
  optimizer.set_optimizableG(network);

  optimizer.optimize(10);
  
}
