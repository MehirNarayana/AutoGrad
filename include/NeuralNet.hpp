#pragma once
#include <vector>

#include <Value.hpp>
#include <memory>

class Neuron {
public:
    Neuron(int inputShape);
    std::vector<std::shared_ptr<Value>> weights;
    std::vector<std::shared_ptr<Value>> biases;
    std::shared_ptr<Value> forward(std::vector<std::shared_ptr<Value>>& inputValues);
    std::vector<std::shared_ptr<Value>> parameters();

private:
    int inputShape;
};

class Layer {
public:
    Layer(int inputShape, int outputShape);
    int inputShape;
    int outputShape;
    std::vector<Neuron> outputNeurons;
    std::vector<std::shared_ptr<Value>> outputValues;
    std::vector<std::shared_ptr<Value>> forward(std::vector<std::shared_ptr<Value>>& inputValues);
    std::vector<std::shared_ptr<Value>> parameters();
};

class Mlp {
public:
    Mlp(int inputShape, std::vector<int> layerShapes);
    int inputShape;
    size_t numLayers;
    std::vector<Layer> layers;
    std::shared_ptr<Value> forward(std::vector<double> inputValues);
    std::vector<std::shared_ptr<Value>> parameters();
};