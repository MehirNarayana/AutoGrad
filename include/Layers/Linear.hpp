#pragma once

#include "Tensor.hpp"
#include <optional>
#include <random>
#include <Layers/Layer.hpp>

template<typename scalarType=float>
class Linear: public Layer<scalarType>{
    size_t inputDim;
    size_t outputDim;
    bool bias;

    public:
        Tensor<scalarType> weights;
        std::optional<Tensor<scalarType>> biases;

    Tensor<scalarType> handleWeights(){
        std::vector<scalarType> weightsVector(inputDim*outputDim);
        const scalarType limit = static_cast<scalarType>(std::sqrt(6.0 / static_cast<double>(inputDim + outputDim)));
        static thread_local std::mt19937 generator(std::random_device{}()); //make generator thread safe
        std::uniform_real_distribution<scalarType> distribution(-limit, limit);
        for (scalarType& weight : weightsVector) {
            weight = distribution(generator);
        }
        std::vector<size_t> shape{outputDim, inputDim};
        return Tensor<scalarType>(std::move(weightsVector), std::move(shape));
    }

    std::optional<Tensor<scalarType>> handleBiases(){
        if (bias){
            std::vector<scalarType> biasesVector(outputDim);
            std::vector<size_t> shape{outputDim};
            return Tensor<scalarType>(std::move(biasesVector), std::move(shape));
        }
        return std::nullopt;
    }
    public:
        Linear(size_t inputDim, size_t outputDim, bool bias=true) : inputDim(inputDim), outputDim(outputDim),
        bias{bias},weights(handleWeights()),
        biases(handleBiases()){}

        Tensor<scalarType> forward(Tensor<scalarType> input) override{
            if (bias){
                return input*weights.transpose(0, 1) + biases.value();
            }
            return input*weights.transpose(0,1);
        }

        std::vector<Tensor<scalarType>> parameters() override {
            std::vector<Tensor<scalarType>> result{weights};
            if (biases) {
                result.push_back(*biases);
            }
            return result;
        }
};
