#pragma once

#include <Tensor.hpp>
#include <Layers/Layer.hpp>
template <typename scalarType>
class Softmax: public Layer<scalarType>{
    public:
        Tensor<scalarType> forward(Tensor<scalarType> inputTensor) override{
            return inputTensor.softmax();
        }
};
