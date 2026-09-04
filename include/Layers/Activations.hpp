#pragma once

#include <Tensor.hpp>
#include <Layers/Layer.hpp>

namespace Activations{
template <typename scalarType>
class Tanh: public Layer<scalarType>{
    public:
        Tensor<scalarType> forward(Tensor<scalarType> inputTensor) override{
            return inputTensor.tanh();
        }
};
}