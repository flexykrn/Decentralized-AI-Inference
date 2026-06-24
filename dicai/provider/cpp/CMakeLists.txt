cmake_minimum_required(VERSION 3.20)
project(dicai-inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Include directories
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${gRPC_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Generate protobuf files
set(PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/../../proto)
set(PROTO_FILES
    ${PROTO_SRC_DIR}/inference.proto
    ${PROTO_SRC_DIR}/coordination.proto
)

# Create generated directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Generate protobuf C++ files
foreach(proto_file ${PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    
    add_custom_command(
        OUTPUT 
            ${CMAKE_CURRENT_BINARY_DIR}/generated/${proto_name}.pb.cc
            ${CMAKE_CURRENT_BINARY_DIR}/generated/${proto_name}.pb.h
            ${CMAKE_CURRENT_BINARY_DIR}/generated/${proto_name}.grpc.pb.cc
            ${CMAKE_CURRENT_BINARY_DIR}/generated/${proto_name}.grpc.pb.h
        COMMAND protobuf::protoc
            --grpc_out=${CMAKE_CURRENT_BINARY_DIR}/generated
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/generated
            -I ${PROTO_SRC_DIR}
            --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
            ${proto_file}
        DEPENDS ${proto_file}
        COMMENT "Generating C++ protobuf files for ${proto_name}"
    )
endforeach()

# Collect generated sources
set(GENERATED_SRCS
    ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/generated/inference.grpc.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/generated/coordination.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/generated/coordination.grpc.pb.cc
)

# Source files
set(SOURCES
    src/server.cpp
    ${GENERATED_SRCS}
)

# Executable
add_executable(dicai-inference-server ${SOURCES})

# Link libraries
target_link_libraries(dicai-inference-server
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
)

# Install
install(TARGETS dicai-inference-server DESTINATION bin)
