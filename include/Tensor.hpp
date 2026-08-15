#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vector>

class TensorBase{
    protected:
        size_t dim = 0;
        std::vector<size_t> dataShape;
        std::vector<size_t> stride;
        std::function<void()> backward;

        void fillStride(size_t lastIndex);

        void swapStride(size_t dim1, size_t dim2);
        void swapShape(size_t dim1, size_t dim2);
        TensorBase();
        TensorBase(std::vector<size_t> dataShape);

    public:
        size_t getDim();
        std::vector<size_t> getShape();
};


template <typename scalarType = float> class Tensor : public TensorBase{
    private:
        std::vector<scalarType> data;

        const int tileM = 32;
        const int tileN = 32;
        const int tileK = 32;


        std::vector<scalarType> getData(){
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

        Tensor(std::vector<scalarType> inputVector, std::vector<size_t> inputDimShape): data(std::move(inputVector)),TensorBase(std::move(inputDimShape)){
            fillStride(dim-1);
        }



    public:
        scalarType operator[](size_t index){
            return data[index];
        }

        template <typename inputType>
        Tensor(const inputType& data){
            findShapeAndFlatten(data, 0);
            stride.resize(dim);
            if (dim>0){
                fillStride(dim-1);
            }
            else{
                throw std::runtime_error{"dimension must be greater than 0"};
            }
        }

        Tensor transpose(size_t dim1, size_t dim2){
            Tensor<scalarType> output =  *this;
            //tranposing is equivalent to just switching the coordinates of every element in the matrix so to we just need to swap the stride
            output.swapStride(dim1, dim2);
            output.swapShape(dim1, dim2);
            return output;
        }

        Tensor operator*(Tensor<scalarType>& other){
            int otherDim = static_cast<int>(other.getDim());
            int currDim = static_cast<int>(dim);

            if (otherDim<2 || currDim < 2){
                throw std::runtime_error{"Ensure both tensors are at least 2d"};
            }

            std::vector<size_t> otherShape = other.getShape();
            int batchDim = std::max(otherDim-2, currDim-2);
            std::vector<size_t> newShape(batchDim+2);
            std::vector<size_t> effectiveStrideCurr(batchDim); //we should count broadcasted dims as stride 0 since we will always data from the first index in this dim
            std::vector<size_t> effectiveStrideOther(batchDim); //we should count broadcasted dims as stride 0 since we will always data from the first index in this dim
            int mCurr  = dataShape[currDim-2];
            int nCurr = dataShape[currDim-1];
            int mOther = otherShape[otherDim-2];
            int nOther = otherShape[otherDim-1];

            if (mOther!=nCurr){
                throw std::runtime_error("Based on the shapes of the final two dimensions of each matrix, these two tensors cannot be multiplied");
            }


            newShape[batchDim+1] = nOther;
            newShape[batchDim] = mCurr;

            int currDimIndex = currDim - 3;
            int otherDimIndex = otherDim - 3;
            int totalElements = mCurr * nOther;
            int k = std::max(currDimIndex, otherDimIndex);
            while (currDimIndex >= 0 && otherDimIndex >= 0) {
                size_t a = dataShape[currDimIndex];
                size_t b = otherShape[otherDimIndex];
                if (a==b){
                    newShape[k] = a;
                    totalElements*=a;
                    effectiveStrideCurr[k] = stride[currDimIndex];
                    effectiveStrideOther[k] = other.stride[otherDimIndex];
                }
                else if (a == 1){
                    newShape[k] = b;
                    totalElements*=b;
                    effectiveStrideCurr[k] = 0;
                    effectiveStrideOther[k] = other.stride[otherDimIndex];
                }

                else if (b==1){
                    newShape[k] = a;
                    totalElements*=a;
                    effectiveStrideCurr[k] = stride[currDimIndex];
                    effectiveStrideOther[k] = 0;
                }

                else {
                    throw std::runtime_error(
                        "Tensors cannot be broadcast together"
                    );
                }

                --k;
                --currDimIndex;
                --otherDimIndex;
            }

            while (currDimIndex >= 0) {
                size_t a = dataShape[currDimIndex];

                newShape[k] = a;
                totalElements *= a;

                effectiveStrideCurr[k] = stride[currDimIndex];
                effectiveStrideOther[k] = 0;

                --k;
                --currDimIndex;
            }

            while (otherDimIndex >= 0) {
                size_t b = otherShape[otherDimIndex];

                newShape[k] = b;
                totalElements *= b;

                effectiveStrideCurr[k] = 0;
                effectiveStrideOther[k] = other.stride[otherDimIndex];

                --k;
                --otherDimIndex;
            }

            int elementsPerBatch = mCurr * nOther;
            int totalBatches = totalElements/elementsPerBatch;

            std::vector<scalarType> result(totalElements);
            size_t currRowStride = stride[dim - 2];
            size_t currColStride = stride[dim - 1];

            size_t otherRowStride = other.stride[otherDim - 2];
            size_t otherColStride = other.stride[otherDim - 1];

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
                int batchEndInex = batchStartIndex+elementsPerBatch;
                int mNew = newShape[batchDim];
                int nNew = newShape[batchDim + 1];
                int kNew = nCurr;
                for (int currVertical = 0; currVertical<mNew; currVertical+=tileM){
                    for (int otherHorizontal= 0; otherHorizontal<nNew; otherHorizontal+=tileN){
                        for (int k=0; k<kNew; k+=tileK){
                            int currHorizontal = k;
                            int otherVertical = k;
                            int endCurrHorizontal = std::min(k+tileK, kNew);
                            int endCurrVertical = std::min(currVertical+tileM, mNew);
                            int endOtherHorizontal = std::min(otherHorizontal+tileN, nNew);

                            for (int y=currVertical; y<endCurrVertical; y++){
                                for (int xOther=otherHorizontal; xOther<endOtherHorizontal; xOther++){
                                    scalarType accum = 0;
                                    for (int index=0; index<endCurrHorizontal-currHorizontal; index++){
                                        int actualCurrCoordinate = currCoordinate + y*currRowStride + (currHorizontal+index) * currColStride;
                                        int actualOtherCoordinate = otherCoordinate + (otherVertical+index) * otherRowStride + xOther * otherColStride;
                                        scalarType multipliedResult = data[actualCurrCoordinate] * other[actualOtherCoordinate];
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

        return Tensor{std::move(result), std::move(newShape)};

        }
};
