#include "gguf_parser.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: GGUFParser <file_path>\n";
        return 1;
    }

    try {
        GGUFParser parser(argv[1]);
        parser.parse();
        parser.print();
    } catch (const GGUFParseError &e) {
        std::cerr << "Error parsing GGUF file: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
