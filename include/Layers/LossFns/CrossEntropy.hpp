#pragma once

#include <Tensor.hpp>

namespace Loss {

class CrossEntropy {
public:
    template <typename scalarType, typename anyType>
    Tensor<scalarType> forward(Tensor<scalarType>& prediction, Tensor<anyType>& target) {
        Tensor<scalarType> probabilities{prediction.softmax()};
        return probabilities.NLLLoss(target);
    }
};
} // namespace Loss
