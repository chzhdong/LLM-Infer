#ifndef GGUF_PARSER_H
#define GGUF_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdint>

class GGUFParseError : public std::runtime_error {
public:
    GGUFParseError(const std::string &message) : std::runtime_error(message) {}
};

class GGUFParser {
public:
    GGUFParser(const std::string &file_path);
    void parse();
    std::vector<std::vector<uint8_t>> loadTensors();
    void print() const;

private:
    std::string file_path_;
    std::string magic_number_;
    uint32_t version_;
    std::map<std::string, std::string> metadata_;
    std::vector<std::map<std::string, std::string>> tensors_info_;
    uint64_t alignment_ = 1;

    static const std::string GGUF_MAGIC_NUMBER;
    static const std::map<int, std::string> VALUE_FORMATS;
    static const std::map<int, std::string> TENSOR_TYPES;

    std::string readString(std::ifstream &file);
    std::pair<std::string, std::string> readMetadataKV(std::ifstream &file);
    std::map<std::string, std::string> readTensorInfo(std::ifstream &file);

    std::string readValue(std::ifstream &file, uint32_t value_type);
    std::vector<uint64_t> readDimensions(std::ifstream &file, uint32_t n_dimensions);
};

#endif // GGUF_PARSER_H
