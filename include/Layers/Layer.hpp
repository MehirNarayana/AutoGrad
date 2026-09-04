#include <Tensor.hpp>
#include <vector>

#pragma once
template <typename scalarType=float>
class Layer{
    public:
        virtual ~Layer() = default;
        virtual Tensor<scalarType> forward(Tensor<scalarType> input) = 0;
        virtual std::vector<Tensor<scalarType>> parameters() {
            return {};
        }
};