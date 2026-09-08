#pragma once

#include <Layers/Layer.hpp>
#include <Tensor.hpp>
template <typename scalarType>
class Softmax : public Layer<scalarType> {
public:
    Tensor<scalarType> forward(Tensor<scalarType> inputTensor) override {
        return inputTensor.softmax();
    }

    void saveLayer(ModelWriter& writer) override {
        writer.saveLayer(std::string{"Softmax"});
        writer.writeNumber(static_cast<uint8_t>(dType<scalarType>()));
    }
};
