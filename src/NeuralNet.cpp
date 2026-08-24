#include "NeuralNet.hpp"
#include <random>

Neuron::Neuron(int inputShape) : inputShape(inputShape), weights(inputShape), biases(1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for (int i = 0; i < inputShape; ++i) {
        weights[i] = std::make_shared<Value>(dis(gen));
        biases[i] = std::make_shared<Value>(dis(gen));
    }
}

std::shared_ptr<Value> Neuron::forward(std::vector<std::shared_ptr<Value>>& inputValues) {
    std::shared_ptr<Value> output = (*weights[0]) * inputValues[0];
    for (int i = 1; i < inputShape; i++) {
        output = *((*weights[i]) * inputValues[i]) + output;
    }
    output = *output + biases[0];
    output->tanh();
    return output;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> temp = weights;
    temp.insert(temp.end(), biases.begin(), biases.end());
    return temp;
}

Layer::Layer(int inputShape, int outputShape) : inputShape(inputShape), outputShape(outputShape) {
    for (int i = 0; i < outputShape; i++) {
        outputNeurons.push_back(Neuron(inputShape));
    }
}

std::vector<std::shared_ptr<Value>>
Layer::forward(std::vector<std::shared_ptr<Value>>& inputValues) {
    outputValues.clear();
    for (int j = 0; j < outputShape; j++) {
        outputValues.push_back(outputNeurons[j].forward(inputValues));
    }
    return outputValues;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {

    std::vector<std::shared_ptr<Value>> layerParams;

    for (auto& neuron : outputNeurons) {
        for (auto& p : neuron.parameters()) {
            layerParams.push_back(p);
        }
    }
    return layerParams;
}

Mlp::Mlp(int inputShape, std::vector<int> layerShapes) : inputShape(inputShape) {
    numLayers = layerShapes.size();
    layers.push_back(Layer(inputShape, layerShapes[0]));

    for (size_t i = 1; i < numLayers; i++) {
        layers.push_back(Layer(layerShapes[i - 1], layerShapes[i]));
    }
}

std::shared_ptr<Value> Mlp::forward(std::vector<double> inputValues) {
    std::vector<std::shared_ptr<Value>> valueConverted;
    for (size_t i = 0; i < inputValues.size(); i++) {
        std::shared_ptr<Value> curr = std::make_shared<Value>(inputValues[i]);
        valueConverted.push_back(curr);
    }

    for (size_t i = 0; i < numLayers; i++) {
        valueConverted = layers[i].forward(valueConverted);
    }
    return valueConverted[0];
}

std::vector<std::shared_ptr<Value>> Mlp::parameters() {

    std::vector<std::shared_ptr<Value>> mlpParams;

    for (auto& layer : layers) {
        for (auto& p : layer.parameters()) {
            mlpParams.push_back(p);
        }
    }
    return mlpParams;
}
