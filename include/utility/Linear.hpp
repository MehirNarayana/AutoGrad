#include "Tensor.hpp"
#include <optional>

template<typename scalarType=float>
class Linear{
    size_t inputDim;
    size_t outputDim;
    bool bias;
    Tensor<scalarType> weights;
    std::optional<Tensor<scalarType>> biases;

    Tensor<scalarType> handleWeights(){
        std::vector<scalarType> weightsVector(inputDim*outputDim);
        std::vector<size_t> shape{outputDim, inputDim};
        return Tensor<scalarType>(std::move(weightsVector), std::move(shape));
    }

    std::optional<Tensor<scalarType>> handleBiases(){
        if (bias){
            std::vector<scalarType> biasesVector(outputDim);
            std::vector<size_t> shape{outputDim};
            return Tensor<scalarType>(std::move(biasesVector), std::move(shape));
        }
        return std::nullopt;
    }
    public:
        Linear(size_t inputDim, size_t outputDim, bool bias=true) : inputDim(inputDim), outputDim(outputDim),
        bias{bias},weights(handleWeights()),
        biases(handleBiases()){}

        Tensor<scalarType> forward(Tensor<scalarType> input){
            if (bias){
                return input*weights.transpose(0, 1) + biases.value();
            }
            return input*weights.transpose(0,1);
        }
};