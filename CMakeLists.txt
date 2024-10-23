cmake_minimum_required(VERSION 3.10)
project(GGUFParser)

set(CMAKE_CXX_STANDARD 17)

# Add the library, specifying the location of source and header files
add_library(Library STATIC
    gguf/gguf_parser.h
    gguf/gguf_parser.cpp
)

target_include_directories(Library PUBLIC ${PROJECT_SOURCE_DIR}/gguf/)

# Define the executable
add_executable(GGUFParser main.cpp)
target_link_libraries(GGUFParser PRIVATE Library)
