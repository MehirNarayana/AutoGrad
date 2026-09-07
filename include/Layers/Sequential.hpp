#pragma once
#include <Tensor.hpp>
#include <Layers/Layer.hpp>
#include <memory>

template <typename scalarType=float>
class Sequential : public Layer<scalarType> {
    private:
        std::vector<std::unique_ptr<Layer<scalarType>>> layers;
    public:
        template <typename layerType>
        void add(layerType &&layer){
            using decayedType = std::decay_t<layerType>;

            static_assert(
                std::is_base_of_v<Layer<scalarType>, decayedType>,
                "Every Sequential argument must be a layer"
            );

            layers.push_back(std::make_unique<decayedType>(std::forward<layerType>(layer)));//move or copy depending on if layer is a lvalue or rvalue
        }

        template <typename ...layerTypes>
        explicit Sequential(layerTypes&&... inputLayers) {
            (add(std::forward<layerTypes>(inputLayers)), ...);
        }

        Tensor<scalarType> forward(Tensor<scalarType> input) override{
            for (std::unique_ptr<Layer<scalarType>> &layer : layers){
                input = layer->forward(std::move(input));
            }

            return input;
       }

        std::vector<Tensor<scalarType>> parameters() override {
            std::vector<Tensor<scalarType>> result;
            for (const std::unique_ptr<Layer<scalarType>>& layer : layers) {
                std::vector<Tensor<scalarType>> layerParameters = layer->parameters();
                result.insert(result.end(), layerParameters.begin(), layerParameters.end());
            }
            return result;
        }
};