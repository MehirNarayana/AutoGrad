#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class ModelWriter {
    std::ofstream file;

public:
    template <typename scalarType>
    void writeVector(const std::vector<scalarType>& inputVector) {
        size_t inputVectorLen = inputVector.size();
        file.write(reinterpret_cast<const char*>(&inputVectorLen), sizeof(inputVectorLen));
        file.write(reinterpret_cast<const char*>(inputVector.data()),
                   inputVector.size() * sizeof(scalarType));
    }

    void writeString(const std::string& inputString) {
        uint32_t inputStringLen = inputString.size();
        file.write(reinterpret_cast<const char*>(&inputStringLen), sizeof(inputStringLen));
        file.write(inputString.data(), inputStringLen);
    }

    void writeBool(const bool& inputBool) {
        uint8_t encodedBool = inputBool ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&encodedBool), sizeof(encodedBool));
    }

    template <typename scalarType>
    void writeNumber(const scalarType number) {
        file.write(reinterpret_cast<const char*>(&number), sizeof(scalarType));
    }

    explicit ModelWriter(const std::string& filePath)
        : file{filePath, std::ios::binary | std::ios::trunc} {
        if (!file.is_open()) {
            throw std::runtime_error{"Could not open model file: " + filePath};
        }
    }

    void saveLayer(std::string layerType) {
        writeString(layerType);
    }

    template <typename scalarType>
    void saveParameters(const std::vector<scalarType>& data, const std::vector<size_t>& dimShape) {
        writeVector<scalarType>(data);
        writeVector<size_t>(dimShape);
    }
};
