#include <Tensor.hpp>
#include <Layers/Softmax.hpp>

namespace Loss{

    class CrossEntropy{
        public:
            template <typename scalarType, typename anyType>
            Tensor<scalarType> forward(Tensor<scalarType> &prediction, Tensor<anyType> &target){
                Tensor<scalarType> predictionLogits{prediction.softmax()};
                return predictionLogits.NLLLoss(target);
            }
    };
}