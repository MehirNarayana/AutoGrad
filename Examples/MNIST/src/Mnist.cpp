#include <Layers/Activations.hpp>
#include <Layers/Linear.hpp>
#include <Layers/LossFns/CrossEntropy.hpp>
#include <Layers/Sequential.hpp>
#include <ModelReader.hpp>
#include <ModelWriter.hpp>
#include <Optimizer.hpp>
#include <Tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AUTOGRAD_MNIST_DATA_DIRECTORY
#define AUTOGRAD_MNIST_DATA_DIRECTORY "Examples/MNIST/dataset"
#endif

namespace {

constexpr size_t imageSize = 28 * 28;
constexpr size_t numberOfClasses = 10;
constexpr size_t hiddenSize = 128;
constexpr size_t trainingEpochs = 5;
constexpr size_t trainingBatchSize = 64;
constexpr float trainingLearningRate = 0.05F;

std::filesystem::path defaultModelPath() {
    return std::filesystem::path{AUTOGRAD_MNIST_DATA_DIRECTORY}.parent_path() / "mnist.model";
}

struct MnistDataset {
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;

    size_t size() const {
        return labels.size();
    }
};

struct Batch {
    Tensor<float> images;
    Tensor<std::int64_t> labels;
};

std::uint32_t readBigEndianUint32(std::ifstream& file) {
    std::array<std::uint8_t, 4> bytes;
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

MnistDataset loadDataset(const std::filesystem::path& imagePath,
                         const std::filesystem::path& labelPath) {
    std::ifstream imageFile(imagePath, std::ios::binary);
    std::ifstream labelFile(labelPath, std::ios::binary);
    if (!imageFile || !labelFile) {
        throw std::runtime_error{"Could not open the MNIST dataset"};
    }

    readBigEndianUint32(imageFile);
    size_t imageCount = readBigEndianUint32(imageFile);
    size_t rows = readBigEndianUint32(imageFile);
    size_t columns = readBigEndianUint32(imageFile);
    readBigEndianUint32(labelFile);
    size_t labelCount = readBigEndianUint32(labelFile);

    MnistDataset dataset;
    dataset.images.resize(imageCount * rows * columns);
    dataset.labels.resize(labelCount);
    imageFile.read(reinterpret_cast<char*>(dataset.images.data()), dataset.images.size());
    labelFile.read(reinterpret_cast<char*>(dataset.labels.data()), dataset.labels.size());
    return dataset;
}

Batch makeBatch(const MnistDataset& dataset,
                const std::vector<size_t>& order,
                size_t start,
                size_t requestedBatchSize) {
    size_t batchSize = std::min(requestedBatchSize, dataset.size() - start);
    std::vector<float> images(batchSize * imageSize);
    std::vector<std::int64_t> labels(batchSize);

    for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        size_t datasetIndex = order[start + batchIndex];
        for (size_t pixel = 0; pixel < imageSize; ++pixel) {
            images[batchIndex * imageSize + pixel] =
                static_cast<float>(dataset.images[datasetIndex * imageSize + pixel]) / 255.0F;
        }
        labels[batchIndex] = dataset.labels[datasetIndex];
    }

    return {
        Tensor<float>{std::move(images), {batchSize, imageSize}, false},
        Tensor<std::int64_t>{std::move(labels), {batchSize}, false},
    };
}

size_t predictedClass(const Tensor<float>& output, size_t batchIndex = 0) {
    size_t prediction = 0;
    size_t outputStart = batchIndex * numberOfClasses;
    for (size_t classIndex = 1; classIndex < numberOfClasses; ++classIndex) {
        if (output[outputStart + classIndex] > output[outputStart + prediction]) {
            prediction = classIndex;
        }
    }
    return prediction;
}

size_t countCorrect(const Tensor<float>& output,
                    const MnistDataset& dataset,
                    const std::vector<size_t>& order,
                    size_t start,
                    size_t batchSize) {
    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        if (predictedClass(output, batchIndex) == dataset.labels[order[start + batchIndex]]) {
            ++correct;
        }
    }
    return correct;
}

float evaluate(Layer<float>& model, const MnistDataset& dataset, size_t batchSize) {
    std::vector<size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), size_t{0});
    size_t correct = 0;

    for (size_t start = 0; start < dataset.size(); start += batchSize) {
        size_t currentBatchSize = std::min(batchSize, dataset.size() - start);
        Batch batch = makeBatch(dataset, order, start, batchSize);
        Tensor<float> output = model.forward(std::move(batch.images));
        correct += countCorrect(output, dataset, order, start, currentBatchSize);
    }

    return static_cast<float>(correct) / static_cast<float>(dataset.size());
}

std::vector<float> loadPgm(const std::filesystem::path& imagePath) {
    std::ifstream file(imagePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"Could not open image: " + imagePath.string()};
    }

    std::string format;
    size_t width;
    size_t height;
    int maximumValue;
    file >> format >> width >> height >> maximumValue;

    if ((format != "P5" && format != "P2") || width != 28 || height != 28) {
        throw std::runtime_error{"Inference expects a 28x28 PGM image"};
    }

    std::vector<float> pixels(imageSize);
    if (format == "P5") {
        file.get();
        std::vector<std::uint8_t> bytes(imageSize);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        for (size_t index = 0; index < imageSize; ++index) {
            pixels[index] = static_cast<float>(bytes[index]) / maximumValue;
        }
    } else {
        for (float& pixel : pixels) {
            int value;
            file >> value;
            pixel = static_cast<float>(value) / maximumValue;
        }
    }
    return pixels;
}

void train(const std::filesystem::path& modelPath) {
    const std::filesystem::path datasetDirectory = AUTOGRAD_MNIST_DATA_DIRECTORY;
    std::cout << "Loading MNIST dataset from " << datasetDirectory << "...\n" << std::flush;
    MnistDataset trainingData = loadDataset(datasetDirectory / "train-images-idx3-ubyte",
                                            datasetDirectory / "train-labels-idx1-ubyte");
    MnistDataset testData = loadDataset(datasetDirectory / "t10k-images-idx3-ubyte",
                                        datasetDirectory / "t10k-labels-idx1-ubyte");
    std::cout << "Loaded " << trainingData.size() << " training images and " << testData.size()
              << " test images.\n";

    std::cout << "Creating model...\n";
    Sequential<float> model{
        Linear<float>{imageSize, hiddenSize},
        Activations::Tanh<float>{},
        Linear<float>{hiddenSize, numberOfClasses},
    };
    Optimizers::SGD<float> optimizer(model.parameters(), trainingLearningRate);
    Loss::CrossEntropy lossFunction;

    std::vector<size_t> order(trainingData.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::mt19937 generator(42);
    const size_t totalBatches = (trainingData.size() + trainingBatchSize - 1) / trainingBatchSize;

    std::cout << "Training for " << trainingEpochs << " epochs with " << totalBatches
              << " batches per epoch...\n";

    for (size_t epoch = 0; epoch < trainingEpochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << '/' << trainingEpochs << " started.\n" << std::flush;
        auto startTime = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), generator);
        float totalLoss = 0;
        size_t correct = 0;
        size_t completedBatches = 0;

        for (size_t start = 0; start < trainingData.size(); start += trainingBatchSize) {
            size_t currentBatchSize = std::min(trainingBatchSize, trainingData.size() - start);
            Batch batch = makeBatch(trainingData, order, start, trainingBatchSize);

            optimizer.zeroGrad();
            Tensor<float> output = model.forward(std::move(batch.images));
            correct += countCorrect(output, trainingData, order, start, currentBatchSize);
            Tensor<float> loss = lossFunction.forward(output, batch.labels);
            totalLoss += loss[0] * currentBatchSize;
            loss.backward();
            optimizer.step();

            ++completedBatches;
            if (completedBatches % 100 == 0 || completedBatches == totalBatches) {
                std::cout << "  completed batch " << completedBatches << '/' << totalBatches << '\n'
                          << std::flush;
            }
        }

        std::cout << "  evaluating test set...\n" << std::flush;
        float testAccuracy = evaluate(model, testData, trainingBatchSize);
        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - startTime;
        std::cout << std::fixed << std::setprecision(4) << "Epoch " << epoch + 1 << '/'
                  << trainingEpochs << "  loss=" << totalLoss / trainingData.size()
                  << "  train_accuracy="
                  << 100.0F * static_cast<float>(correct) / trainingData.size() << '%'
                  << "  test_accuracy=" << 100.0F * testAccuracy << '%'
                  << "  time=" << elapsed.count() << "s\n";
    }

    std::cout << "Saving model to " << modelPath << "...\n" << std::flush;
    ModelWriter writer{modelPath.string()};
    model.saveLayer(writer);
    std::cout << "Saved model to " << modelPath << '\n';
}

void infer(const std::filesystem::path& modelPath, const std::filesystem::path& imagePath) {
    std::cout << "Loading model from " << modelPath << "...\n" << std::flush;
    ModelReader reader{modelPath.string()};
    std::unique_ptr<Layer<float>> model = reader.loadModel<float>();
    std::cout << "Loading image " << imagePath << "...\n" << std::flush;
    Tensor<float> image{loadPgm(imagePath), {1, imageSize}, false};
    Tensor<float> output = model->forward(std::move(image));

    std::cout << "Predicted digit: " << predictedClass(output) << '\n';
}

void printUsage(const char* executable) {
    std::cout << "Usage:\n"
              << "  " << executable << " train [model-file]\n"
              << "  " << executable << " infer <model-file> <28x28-image.pgm>\n\n"
              << "Default training values:\n"
              << "  dataset: " << AUTOGRAD_MNIST_DATA_DIRECTORY << '\n'
              << "  model-file: " << defaultModelPath() << '\n'
              << "  epochs: " << trainingEpochs << '\n'
              << "  batch-size: " << trainingBatchSize << '\n'
              << "  learning-rate: " << trainingLearningRate << '\n';
}

} // namespace

int main(int argumentCount, char** arguments) {
    try {
        if (argumentCount < 2 || std::string{arguments[1]} == "--help") {
            printUsage(arguments[0]);
            return 0;
        }

        std::string mode = arguments[1];
        if (mode == "train") {
            std::filesystem::path modelPath = argumentCount > 2 ? arguments[2] : defaultModelPath();
            train(modelPath);
        } else if (mode == "infer" && argumentCount == 4) {
            infer(arguments[2], arguments[3]);
        } else {
            printUsage(arguments[0]);
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "MNIST example failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
