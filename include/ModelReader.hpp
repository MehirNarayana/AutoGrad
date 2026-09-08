#pragma once

#include <Layers/Activations.hpp>
#include <Layers/Layer.hpp>
#include <Layers/Linear.hpp>
#include <Layers/Sequential.hpp>
#include <Layers/Softmax.hpp>
#include <ScalarType.hpp>

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ModelReader {
    std::ifstream file;

    template <typename valueType>
    valueType readNumber() {
        valueType number;
        file.read(reinterpret_cast<char*>(&number), sizeof(valueType));
        return number;
    }

    template <typename valueType>
    std::vector<valueType> readVector() {
        size_t vectorLength = readNumber<size_t>();
        std::vector<valueType> result(vectorLength);
        file.read(reinterpret_cast<char*>(result.data()), vectorLength * sizeof(valueType));
        return result;
    }

    std::string readString() {
        uint32_t stringLength = readNumber<uint32_t>();
        std::string result(stringLength, '\0');
        file.read(result.data(), stringLength);
        return result;
    }

    bool readBool() {
        return readNumber<uint8_t>() == 1;
    }

    template <typename scalarType>
    std::unique_ptr<Layer<scalarType>> loadLinear() {
        std::vector<scalarType> weightsData = readVector<scalarType>();
        std::vector<size_t> weightsShape = readVector<size_t>();
        Tensor<scalarType> weights{std::move(weightsData), std::move(weightsShape), false};

        std::optional<Tensor<scalarType>> biases;
        if (readBool()) {
            std::vector<scalarType> biasData = readVector<scalarType>();
            std::vector<size_t> biasShape = readVector<size_t>();
            biases.emplace(std::move(biasData), std::move(biasShape), false);
        }

        return std::make_unique<Linear<scalarType>>(std::move(weights), std::move(biases));
    }

    template <typename scalarType>
    std::unique_ptr<Layer<scalarType>> loadSequential() {
        size_t layerCount = readNumber<size_t>();
        auto sequential = std::make_unique<Sequential<scalarType>>();

        for (size_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
            sequential->add(loadLayer<scalarType>());
        }
        return sequential;
    }

    template <typename scalarType>
    std::unique_ptr<Layer<scalarType>> loadLayer() {
        std::string layerType = readString();
        DType savedType = static_cast<DType>(readNumber<uint8_t>());

        if (savedType != dType<scalarType>()) {
            throw std::runtime_error{"Requested scalar type does not match the saved model"};
        }

        if (layerType == "Linear") {
            return loadLinear<scalarType>();
        }
        if (layerType == "Tanh") {
            return std::make_unique<Activations::Tanh<scalarType>>();
        }
        if (layerType == "Softmax") {
            return std::make_unique<Softmax<scalarType>>();
        }
        if (layerType == "Sequential") {
            return loadSequential<scalarType>();
        }

        throw std::runtime_error{"Unknown layer type: " + layerType};
    }

public:
    explicit ModelReader(const std::string& filePath) : file(filePath, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error{"Could not open model file: " + filePath};
        }
    }

    template <typename scalarType>
    std::unique_ptr<Layer<scalarType>> loadModel() {
        return loadLayer<scalarType>();
    }
};
