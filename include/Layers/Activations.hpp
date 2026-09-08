#pragma once

#include <Layers/Layer.hpp>
#include <Tensor.hpp>

namespace Activations {
template <typename scalarType>
class Tanh : public Layer<scalarType> {
public:
    Tensor<scalarType> forward(Tensor<scalarType> inputTensor) override {
        return inputTensor.tanh();
    }
    void saveLayer(ModelWriter& writer) override {
        writer.saveLayer(std::string{"Tanh"});
        writer.writeNumber(static_cast<uint8_t>(dType<scalarType>()));
    }
};
} // namespace Activations