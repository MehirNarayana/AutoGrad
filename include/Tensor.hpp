#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

class TensorBase{
    protected:
        size_t dim = 0;
        std::vector<size_t> dataShape;
        std::vector<size_t> stride;

        void fillStride(size_t lastIndex);

        void swapStride(size_t dim1, size_t dim2);
        void swapShape(size_t dim1, size_t dim2);
        TensorBase();
        TensorBase(std::vector<size_t> dataShape);

    public:
        size_t getDim();
        std::vector<size_t> getShape();
        std::function<void()> backward;
        int getNumTotalElements();
};


template <typename scalarType = float>
class Tensor : public TensorBase,
               public std::enable_shared_from_this<Tensor<scalarType>>{
    private:
        std::vector<scalarType> data;
        std::vector<std::shared_ptr<Tensor<scalarType>>> parents;

        const static int tileM = 32;
        const static int tileN = 32;
        const static int tileK = 32;


        std::vector<scalarType> &getData(){
            return data;
        }


        template <typename inputType>
        struct isVector : std::false_type {};

        template <typename inputType, typename Alloc>
        struct isVector<std::vector<inputType, Alloc>> : std::true_type{};

        bool isScalarType(scalarType x){
            return true;
        }

        template <typename randomType>
        bool isScalarType(randomType x){
            return false;
        }


        template <typename inputType>
        size_t findShapeAndFlatten(const inputType& data, int currDepth){
            if constexpr (!isVector<inputType>::value){
                if (isScalarType(data)){
                    this->data.push_back(data);
                    return 1;
                }
                else{
                    throw std::runtime_error{"Data inside tensor does not match type supplied to Tensor"};
                }
            }
            else{
                std::optional<size_t> nextDimSize;
                size_t currDimSize = data.size();
                if (dim <= currDepth){
                    dim++;
                    dataShape.push_back(currDimSize);
                }

                for (size_t i = 0; i<currDimSize; i++){
                    size_t currSize = findShapeAndFlatten(data[i], currDepth+ 1);
                    if (nextDimSize){
                        if (nextDimSize != currSize){
                            throw std::runtime_error{"Not a rectangular tensor"};
                        }
                    }
                    else{
                        nextDimSize = currSize;
                    }
                }
                return currDimSize;
            }
        }

        void broadcast(
            Tensor<scalarType>& other,
            std::vector<size_t>& newShape,
            std::vector<size_t>& effectiveStrideCurr,
            std::vector<size_t>& effectiveStrideOther,
            int& batchDim,
            int& totalElements,
            int ignoredDimensions
        ){
            int otherDim = static_cast<int>(other.getDim());
            int currDim = static_cast<int>(dim);

            if (otherDim < ignoredDimensions || currDim < ignoredDimensions){
                throw std::runtime_error{"Not enough dimensions to broadcast"};
            }

            batchDim = std::max(
                otherDim - ignoredDimensions,
                currDim - ignoredDimensions
            );
            newShape.resize(batchDim);
            effectiveStrideCurr.resize(batchDim);
            effectiveStrideOther.resize(batchDim);
            totalElements = 1;

            std::vector<size_t> otherShape = other.getShape();
            int currDimIndex = currDim - ignoredDimensions - 1;
            int otherDimIndex = otherDim - ignoredDimensions - 1;
            int k = batchDim - 1;

            while (currDimIndex >= 0 && otherDimIndex >= 0) {
                size_t a = dataShape[currDimIndex];
                size_t b = otherShape[otherDimIndex];

                if (a == b){
                    newShape[k] = a;
                    effectiveStrideCurr[k] = stride[currDimIndex];
                    effectiveStrideOther[k] = other.stride[otherDimIndex];
                }
                else if (a == 1){
                    newShape[k] = b;
                    effectiveStrideCurr[k] = 0;
                    effectiveStrideOther[k] = other.stride[otherDimIndex];
                }
                else if (b == 1){
                    newShape[k] = a;
                    effectiveStrideCurr[k] = stride[currDimIndex];
                    effectiveStrideOther[k] = 0;
                }
                else {
                    throw std::runtime_error("Tensors cannot be broadcast together");
                }

                totalElements *= newShape[k];
                --k;
                --currDimIndex;
                --otherDimIndex;
            }

            while (currDimIndex >= 0) {
                newShape[k] = dataShape[currDimIndex];
                effectiveStrideCurr[k] = stride[currDimIndex];
                effectiveStrideOther[k] = 0;
                totalElements *= newShape[k];
                --k;
                --currDimIndex;
            }

            while (otherDimIndex >= 0) {
                newShape[k] = otherShape[otherDimIndex];
                effectiveStrideCurr[k] = 0;
                effectiveStrideOther[k] = other.stride[otherDimIndex];
                totalElements *= newShape[k];
                --k;
                --otherDimIndex;
            }
        }


        std::shared_ptr<Tensor<scalarType>> tiledMatmul(
            const std::shared_ptr<Tensor<scalarType>>& other,
            std::vector<size_t> newShape,
            const std::vector<size_t>& effectiveStrideCurr,
            const std::vector<size_t>& effectiveStrideOther,
            int totalElements,
            int mCurr,
            int nCurr,
            int nOther,
            int batchDim
        ){
            int elementsPerBatch = mCurr * nOther;
            int totalBatches = totalElements/elementsPerBatch;

            std::vector<scalarType> result(totalElements);
            size_t currRowStride = stride[dim - 2];
            size_t currColStride = stride[dim - 1];

            int otherDim = static_cast<int>(other->getDim());
            size_t otherRowStride = other->stride[otherDim - 2];
            size_t otherColStride = other->stride[otherDim - 1];

            for (int batchIndex = 0; batchIndex<totalBatches; batchIndex++){
                int batchIndexCopy = batchIndex;
                int currCoordinate = 0;
                int otherCoordinate = 0;
                for (int index = batchDim-1; index >= 0; index--){
                    int currDimStrideCoefficient = batchIndexCopy % newShape[index];
                    batchIndexCopy/=newShape[index];
                    currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                    otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
                }

                int batchStartIndex = batchIndex*elementsPerBatch;
                int mNew = newShape[batchDim];
                int nNew = newShape[batchDim + 1];
                int kNew = nCurr;
                for (int currVertical = 0; currVertical<mNew; currVertical+=Tensor::tileM){
                    for (int otherHorizontal= 0; otherHorizontal<nNew; otherHorizontal+=Tensor::tileN){
                        for (int k=0; k<kNew; k+=tileK){
                            int currHorizontal = k;
                            int otherVertical = k;
                            int endCurrHorizontal = std::min(k+tileK, kNew);
                            int endCurrVertical = std::min(currVertical+Tensor::tileM, mNew);
                            int endOtherHorizontal = std::min(otherHorizontal+Tensor::tileN, nNew);

                            for (int y=currVertical; y<endCurrVertical; y++){
                                for (int xOther=otherHorizontal; xOther<endOtherHorizontal; xOther++){
                                    scalarType accum = 0;
                                    for (int index=0; index<endCurrHorizontal-currHorizontal; index++){
                                        int actualCurrCoordinate = currCoordinate + y*currRowStride + (currHorizontal+index) * currColStride;
                                        int actualOtherCoordinate = otherCoordinate + (otherVertical+index) * otherRowStride + xOther * otherColStride;
                                        scalarType multipliedResult = data[actualCurrCoordinate] * (*other)[actualOtherCoordinate];
                                        accum+=multipliedResult;
                                    }
                                    int targetCoordinate = batchStartIndex + y * nNew + xOther;
                                    result[targetCoordinate] += accum;
                                }
                            }
                        }
                    }
                }
            }

            bool outputTracksGradient = trackGradient || other->trackGradient;
            std::shared_ptr<Tensor<scalarType>> output(
                new Tensor<scalarType>(
                    std::move(result),
                    std::move(newShape),
                    outputTracksGradient
                )
            );
            if (outputTracksGradient){
                output->parents = {this->shared_from_this(), other};
            }
            std::weak_ptr<Tensor<scalarType>> currWeak{this->shared_from_this()};
            std::weak_ptr<Tensor<scalarType>> otherWeak{other};
            std::weak_ptr<Tensor<scalarType>> outputWeak{output};


            output->backward = [batchDim, currWeak, otherWeak, outputWeak, effectiveStrideCurr = std::move(effectiveStrideCurr), effectiveStrideOther=std::move(effectiveStrideOther), totalBatches, elementsPerBatch](){
                std::shared_ptr<Tensor<scalarType>> output = outputWeak.lock();
                std::shared_ptr<Tensor<scalarType>> curr = currWeak.lock();
                std::shared_ptr<Tensor<scalarType>> other = otherWeak.lock();

                if (!output || !curr || !other) {
                    return;
                }

                std::vector<scalarType> &outputGradientVector = output->gradient->getData();
                size_t outputRowStride = output->stride[output->dim - 2];
                size_t outputColStride = output->stride[output->dim - 1];

                size_t currRowStride = curr->stride[curr->dim - 2];
                size_t currColStride = curr->stride[curr->dim - 1];

                size_t otherRowStride = other->stride[other->dim - 2];
                size_t otherColStride = other->stride[other->dim - 1];

                for (int batchIndex = 0; batchIndex<totalBatches; batchIndex++){
                    int batchIndexCopy = batchIndex;
                    int currCoordinate = 0;
                    int otherCoordinate = 0;

                    for (int index = batchDim-1; index >= 0; index--){
                        int currDimStrideCoefficient = batchIndexCopy % output->dataShape[index];
                        batchIndexCopy/=output->dataShape[index];
                        currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                        otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
                    }


                    int batchStartIndex = batchIndex*elementsPerBatch;

                    if (curr->trackGradient){
                        std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
                        int mNew = curr->dataShape[curr->dim-2];
                        int nNew = curr->dataShape[curr->dim-1];
                        int kNew = other->dataShape[other->dim-1]; //since we want the tranpose of other

                        for (int outputVertical = 0; outputVertical<mNew; outputVertical+=Tensor::tileM){
                            for (int otherHorizontal = 0; otherHorizontal<nNew; otherHorizontal+=Tensor::tileN){
                                for (int k=0; k<kNew; k+=tileK){
                                    int outputHorizontal = k;
                                    int otherVertical = k;
                                    int endOutputHorizontal = std::min(k+tileK, kNew);
                                    int endOutputVertical = std::min(outputVertical+Tensor::tileM, mNew);
                                    int endOtherHorizontal = std::min(otherHorizontal+Tensor::tileN, nNew);

                                    for (int y=outputVertical; y<endOutputVertical; y++){
                                        for (int xOther=otherHorizontal; xOther<endOtherHorizontal; xOther++){
                                            scalarType accum = 0;
                                            for (int index=0; index<endOutputHorizontal-outputHorizontal; index++){
                                                int actualOutputCoordinate = batchStartIndex + y*outputRowStride + (outputHorizontal + index) * outputColStride;
                                                int actualOtherCoordinate = otherCoordinate + xOther * otherRowStride + (otherVertical+index) * otherColStride; //inverted cuz transposed
                                                scalarType multipliedResult = outputGradientVector[actualOutputCoordinate] * other->data[actualOtherCoordinate];
                                                accum+=multipliedResult;
                                            }
                                            int actualCurrCoordinate = currCoordinate + y*currRowStride + xOther * currColStride;
                                            currGradientVector[actualCurrCoordinate] += accum;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (other->trackGradient){
                        std::vector<scalarType>& otherGradientVector = other->ensureGradient().getData();
                        int mNew = other->dataShape[other->dim-2];
                        int nNew = other->dataShape[other->dim-1];
                        int kNew = curr->dataShape[curr->dim-2]; //since we want the tranpose of curr

                        for (int currVertical = 0; currVertical<mNew; currVertical+=Tensor::tileM){
                            for (int outputHorizontal = 0; outputHorizontal<nNew; outputHorizontal+=Tensor::tileN){
                                for (int k=0; k<kNew; k+=tileK){
                                    int currHorizontal = k;
                                    int outputVertical = k;
                                    int endCurrHorizontal = std::min(k+tileK, kNew);
                                    int endCurrVertical = std::min(currVertical+Tensor::tileM, mNew);
                                    int endOutputHorizontal = std::min(outputHorizontal+Tensor::tileN, nNew);

                                    for (int y=currVertical; y<endCurrVertical; y++){
                                        for (int xCurr=outputHorizontal; xCurr<endOutputHorizontal; xCurr++){
                                            scalarType accum = 0;
                                            for (int index=0; index<endCurrHorizontal-currHorizontal; index++){
                                                int actualOutputCoordinate = batchStartIndex + (outputVertical + index)*outputRowStride + xCurr * outputColStride;
                                                int actualCurrCoordinate = currCoordinate + (currHorizontal + index) * currRowStride + y * currColStride; //inverted cuz transposed
                                                scalarType multipliedResult = outputGradientVector[actualOutputCoordinate] * curr->data[actualCurrCoordinate];
                                                accum+=multipliedResult;
                                            }
                                            int actualOtherCoordinate = otherCoordinate + y*otherRowStride + xCurr * otherColStride;
                                            otherGradientVector[actualOtherCoordinate] += accum;
                                        }
                                    }
                                }
                            }
                        }
                    }

                }



            };

            return output;
        }

        Tensor(
            std::vector<scalarType> inputVector,
            std::vector<size_t> inputDimShape,
            bool shouldTrackGradient
        ): TensorBase(std::move(inputDimShape)),
           data(std::move(inputVector)),
           trackGradient(shouldTrackGradient){
            fillStride(dim-1);
        }



    public:
        bool trackGradient = true;
        std::unique_ptr<Tensor<scalarType>> gradient;

        // Use this for tensors that will participate in autograd.  It creates
        // the shared owner required by shared_from_this() in tensor operators.
        template <typename inputType>
        static std::shared_ptr<Tensor<scalarType>> create(
            const inputType& data,
            bool shouldTrackGradient = true
        ){
            return std::shared_ptr<Tensor<scalarType>>(
                new Tensor<scalarType>(data, shouldTrackGradient)
            );
        }

        Tensor(const Tensor& other)
            : TensorBase(other),
              data(other.data),
              parents(other.parents),
              trackGradient(other.trackGradient){
            if (other.gradient){
                gradient = std::unique_ptr<Tensor<scalarType>>(
                    new Tensor<scalarType>(*other.gradient)
                );
            }
        }

        Tensor& operator=(const Tensor& other){
            if (this == &other){
                return *this;
            }

            TensorBase::operator=(other);
            data = other.data;
            parents = other.parents;
            trackGradient = other.trackGradient;

            if (other.gradient){
                gradient = std::unique_ptr<Tensor<scalarType>>(
                    new Tensor<scalarType>(*other.gradient)
                );
            }
            else{
                gradient.reset();
            }

            return *this;
        }

        Tensor<scalarType>& ensureGradient(){
            if (!gradient){
                std::vector<scalarType> zeros(data.size(), scalarType{});
                gradient = std::unique_ptr<Tensor<scalarType>>(
                    new Tensor<scalarType>(
                        std::move(zeros),
                        dataShape,
                        false
                    )
                );
            }

            return *gradient;
        }

        scalarType operator[](size_t index){
            return data[index];
        }

    private:

        template <typename inputType>
        Tensor(const inputType& data, bool shouldTrackGradient = true)
            : trackGradient(shouldTrackGradient){
            findShapeAndFlatten(data, 0);
            stride.resize(dim);
            if (dim>0){
                fillStride(dim-1);
            }
            else{
                throw std::runtime_error{"dimension must be greater than 0"};
            }
        }

    public:

        Tensor transpose(size_t dim1, size_t dim2){
            Tensor<scalarType> output =  *this;
            //tranposing is equivalent to just switching the coordinates of every element in the matrix so to we just need to swap the stride
            output.swapStride(dim1, dim2);
            output.swapShape(dim1, dim2);
            return output;
        }

        std::shared_ptr<Tensor<scalarType>> operator*(
            const std::shared_ptr<Tensor<scalarType>>& other
        ){
            if (!other){
                throw std::runtime_error{"Cannot multiply by a null tensor"};
            }

            int otherDim = static_cast<int>(other->getDim());
            int currDim = static_cast<int>(dim);

            if (otherDim<2 || currDim < 2){
                throw std::runtime_error{"Ensure both tensors are at least 2d"};
            }

            std::vector<size_t> otherShape = other->getShape();
            int mCurr  = dataShape[currDim-2];
            int nCurr = dataShape[currDim-1];
            int mOther = otherShape[otherDim-2];
            int nOther = otherShape[otherDim-1];

            if (mOther!=nCurr){
                throw std::runtime_error("Based on the shapes of the final two dimensions of each matrix, these two tensors cannot be multiplied");
            }


            int batchDim;
            int totalElements;
            std::vector<size_t> newShape;
            std::vector<size_t> effectiveStrideCurr;
            std::vector<size_t> effectiveStrideOther;
            broadcast(
                *other,
                newShape,
                effectiveStrideCurr,
                effectiveStrideOther,
                batchDim,
                totalElements,
                2
            );
            totalElements *= mCurr * nOther;

            newShape.push_back(mCurr);
            newShape.push_back(nOther);

            return tiledMatmul(
                other,
                std::move(newShape),
                effectiveStrideCurr,
                effectiveStrideOther,
                totalElements,
                mCurr,
                nCurr,
                nOther,
                batchDim
            );

        }
};
