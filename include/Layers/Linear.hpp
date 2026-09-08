#pragma once

#include "Tensor.hpp"
#include <Layers/Layer.hpp>
#include <ModelWriter.hpp>
#include <ScalarType.hpp>
#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>

template <typename scalarType = float>
class Linear : public Layer<scalarType> {
    size_t inputDim;
    size_t outputDim;
    bool bias;
    bool trackGradient;

public:
    Tensor<scalarType> weights;
    std::optional<Tensor<scalarType>> biases;

    Tensor<scalarType> handleWeights() {
        static_assert(isSupportedFloatingPointScalarType<scalarType>,
                      "Linear requires a floating-point scalar type");

        std::vector<scalarType> weightsVector(inputDim * outputDim);
        const scalarType limit =
            static_cast<scalarType>(std::sqrt(6.0 / static_cast<double>(inputDim + outputDim)));
        static std::mt19937 generator(42);
        std::uniform_real_distribution<scalarType> distribution(-limit, limit);
        for (scalarType& weight : weightsVector) {
            weight = distribution(generator);
        }
        std::vector<size_t> shape{outputDim, inputDim};
        return Tensor<scalarType>(std::move(weightsVector), std::move(shape));
    }

    std::optional<Tensor<scalarType>> handleBiases() {
        if (bias) {
            std::vector<scalarType> biasesVector(outputDim);
            std::vector<size_t> shape{outputDim};
            return Tensor<scalarType>(std::move(biasesVector), std::move(shape));
        }
        return std::nullopt;
    }

public:
    Linear(size_t inputDim, size_t outputDim, bool bias = true)
        : inputDim(inputDim), outputDim(outputDim), bias{bias}, weights(handleWeights()),
          biases(handleBiases()) {}

    Linear(Tensor<scalarType> inputWeights,
           std::optional<Tensor<scalarType>> inputBiases = std::nullopt)
        : inputDim(0), outputDim(0), bias(inputBiases.has_value()),
          weights(std::move(inputWeights)), biases(std::move(inputBiases)) {
        const std::vector<size_t>& weightShape = weights.getShape();
        if (weightShape.size() != 2) {
            throw std::runtime_error{"Linear weights must have two dimensions"};
        }

        outputDim = weightShape[0];
        inputDim = weightShape[1];

        if (bias) {
            const std::vector<size_t>& biasShape = biases->getShape();
            if (biasShape.size() != 1 || biasShape[0] != outputDim) {
                throw std::runtime_error{"Linear bias shape must match the output dimension"};
            }
        }
    }

    Tensor<scalarType> forward(Tensor<scalarType> input) override {
        if (bias) {
            return input * weights.transpose(0, 1) + biases.value();
        }
        return input * weights.transpose(0, 1);
    }

    std::vector<Tensor<scalarType>> parameters() override {
        std::vector<Tensor<scalarType>> result{weights};
        if (biases) {
            result.push_back(*biases);
        }
        return result;
    }

    void saveLayer(ModelWriter& writer) override {
        writer.saveLayer(std::string{"Linear"});
        writer.writeNumber(static_cast<uint8_t>(dType<scalarType>()));
        writer.saveParameters(weights.getData(), weights.getShape());
        writer.writeBool(bias);

        if (bias) {
            writer.saveParameters((*biases).getData(), (*biases).getShape());
        }
    }
};
