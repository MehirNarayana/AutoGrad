#pragma once

#include <Tensor.hpp>

namespace Optimizers {

template <typename scalarType>
class SGD {
    std::vector<Tensor<scalarType>> parameters;
    scalarType learningRate;
    public:
        SGD(std::vector<Tensor<scalarType>> parameters, scalarType learningRate)
            : parameters(parameters), learningRate(learningRate) {};

        void step() {
            for (Tensor<scalarType> parameter : parameters) {
                parameter.step(learningRate);
            }
        }

        void zeroGrad() {
            for (Tensor<scalarType> &parameter : parameters) {
                parameter.zeroGrad();
            }
        }
};
}