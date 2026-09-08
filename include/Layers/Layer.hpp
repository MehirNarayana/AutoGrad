#pragma once

#include <ModelWriter.hpp>
#include <Tensor.hpp>
#include <vector>

template <typename scalarType = float>
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor<scalarType> forward(Tensor<scalarType> input) = 0;
    virtual std::vector<Tensor<scalarType>> parameters() {
        return {};
    }

    virtual void saveLayer(ModelWriter& writer) = 0;
};
