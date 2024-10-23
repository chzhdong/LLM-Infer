#include "gguf_parser.h"

const std::string GGUFParser::GGUF_MAGIC_NUMBER = "GGUF";
const std::map<int, std::string> GGUFParser::VALUE_FORMATS = {
    {0, "B"}, {1, "b"}, {2, "H"}, {3, "h"}, {4, "I"}, {5, "i"},
    {6, "f"}, {7, "?"}, {10, "Q"}, {11, "q"}, {12, "d"}
};

const std::map<int, std::string> GGUFParser::TENSOR_TYPES = {
    {0, "GGML_TYPE_F32"}, {1, "GGML_TYPE_F16"}, {2, "GGML_TYPE_Q4_0"},
    {3, "GGML_TYPE_Q4_1"}, {6, "GGML_TYPE_Q5_0"}, {7, "GGML_TYPE_Q5_1"},
    {8, "GGML_TYPE_Q8_0"}, {9, "GGML_TYPE_Q8_1"}, {10, "GGML_TYPE_Q2_K"},
    {11, "GGML_TYPE_Q3_K"}, {12, "GGML_TYPE_Q4_K"}, {13, "GGML_TYPE_Q5_K"},
    {14, "GGML_TYPE_Q6_K"}, {15, "GGML_TYPE_Q8_K"}
};

// Constructor: Initialize the file path
GGUFParser::GGUFParser(const std::string &file_path) : file_path_(file_path) {}

// Read a string from the file
std::string GGUFParser::readString(std::ifstream &file) {
    uint64_t length;
    file.read(reinterpret_cast<char *>(&length), sizeof(length));
    std::string str(length, '\0');
    file.read(&str[0], length);
    return str;
}

// Read a metadata key-value pair
std::pair<std::string, std::string> GGUFParser::readMetadataKV(std::ifstream &file) {
    std::string key = readString(file);
    uint32_t value_type;
    file.read(reinterpret_cast<char *>(&value_type), sizeof(value_type));
    std::string value = readValue(file, value_type);
    return {key, value};
}

// Read a value from the file based on the value_type
std::string GGUFParser::readValue(std::ifstream &file, uint32_t value_type) {
    switch (value_type) {
    case 0: { // UINT8
        uint8_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 1: { // INT8
        int8_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 2: { // UINT16
        uint16_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 3: { // INT16
        int16_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 4: { // UINT32
        uint32_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 5: { // INT32
        int32_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 6: { // FLOAT32
        float val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 7: { // BOOL (True or False)
        bool val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return val ? "true" : "false";
    }
    case 10: { // UINT64
        uint64_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 11: { // INT64
        int64_t val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 12: { // FLOAT64
        double val;
        file.read(reinterpret_cast<char *>(&val), sizeof(val));
        return std::to_string(val);
    }
    case 8: // STRING
        return readString(file);
    case 9: { // ARRAY
        uint32_t array_type;
        uint64_t array_len;
        file.read(reinterpret_cast<char *>(&array_type), sizeof(array_type));
        file.read(reinterpret_cast<char *>(&array_len), sizeof(array_len));
        std::vector<std::string> values;
        for (uint64_t i = 0; i < array_len; ++i) {
            values.push_back(readValue(file, array_type));
        }
        return "[" + std::to_string(array_len) + " elements]";
    }
    default:
        std::cerr << "Unsupported value type: " << value_type << std::endl;
        throw GGUFParseError("Unsupported value type: " + std::to_string(value_type));
    }
}

// Read tensor information
std::map<std::string, std::string> GGUFParser::readTensorInfo(std::ifstream &file) {
    std::map<std::string, std::string> tensor_info;
    tensor_info["name"] = readString(file);
    uint32_t n_dimensions;
    file.read(reinterpret_cast<char *>(&n_dimensions), sizeof(n_dimensions));
    tensor_info["n_dimensions"] = std::to_string(n_dimensions);

    std::vector<uint64_t> dimensions = readDimensions(file, n_dimensions);
    tensor_info["dimensions"] = "[" + std::to_string(dimensions.size()) + " elements]";

    uint32_t tensor_type;
    file.read(reinterpret_cast<char *>(&tensor_type), sizeof(tensor_type));
    tensor_info["type"] = std::to_string(tensor_type);

    uint64_t offset;
    file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    tensor_info["offset"] = std::to_string(offset);
    return tensor_info;
}

// Read the dimensions of a tensor
std::vector<uint64_t> GGUFParser::readDimensions(std::ifstream &file, uint32_t n_dimensions) {
    std::vector<uint64_t> dimensions(n_dimensions);
    file.read(reinterpret_cast<char *>(dimensions.data()), n_dimensions * sizeof(uint64_t));
    return dimensions;
}

// Parse the GGUF file
void GGUFParser::parse() {
    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) throw GGUFParseError("Failed to open file");

    // Read magic number
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic, 4) != GGUF_MAGIC_NUMBER) throw GGUFParseError("Invalid magic number");

    // Read version
    file.read(reinterpret_cast<char *>(&version_), sizeof(version_));
    if (version_ != 3) throw GGUFParseError("Unsupported version");

    // Read the number of tensors and metadata key-value pairs
    uint64_t tensor_count, metadata_kv_count;
    file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
    file.read(reinterpret_cast<char *>(&metadata_kv_count), sizeof(metadata_kv_count));

    // Read metadata
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        auto [key, value] = readMetadataKV(file);
        metadata_[key] = value;
    }

    // Read tensor info
    for (uint64_t i = 0; i < tensor_count; ++i) {
        tensors_info_.push_back(readTensorInfo(file));
    }
}

// Print the parsed information
void GGUFParser::print() const {
    std::cout << "Magic Number: GGUF\n";
    std::cout << "Version: " << version_ << "\n";
    
    std::cout << "Metadata:\n";
    for (const auto &kv : metadata_) {
        std::cout << "  " << kv.first << ": " << kv.second << "\n";
    }

    std::cout << "Tensors Info:\n";
    for (const auto &tensor_info : tensors_info_) {
        std::cout << "  Name: " << tensor_info.at("name")
                  << ", Dimensions: " << tensor_info.at("dimensions")
                  << ", Type: " << tensor_info.at("type")
                  << ", Offset: " << tensor_info.at("offset") << "\n";
    }
}
