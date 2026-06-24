#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"
#include "coordination.grpc.pb.h"
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using inference::InferenceService;
using inference::LayerComputeRequest;
using inference::LayerComputeResponse;
using inference::InferenceRequest;
using inference::InferenceResponse;
using inference::Tensor;

namespace dicai {

// InferenceBackend is the abstract base class for all inference backends
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual bool Initialize() = 0;
    virtual bool LoadLayers(int layer_start, int layer_end) = 0;
    virtual bool Forward(const Tensor& input, Tensor* output) = 0;
    virtual bool Unload() = 0;
    virtual std::string GetName() const = 0;
    virtual int GetMemoryUsage() const = 0;
};

// CPUBackend is a mock backend that returns input unchanged
class CPUBackend : public InferenceBackend {
public:
    bool Initialize() override {
        std::cout << "[CPU] Initializing backend..." << std::endl;
        return true;
    }

    bool LoadLayers(int layer_start, int layer_end) override {
        std::cout << "[CPU] Loading layers " << layer_start << "-" << layer_end << std::endl;
        layer_start_ = layer_start;
        layer_end_ = layer_end;
        return true;
    }

    bool Forward(const Tensor& input, Tensor* output) override {
        // Mock computation: return input unchanged
        *output = input;
        std::cout << "[CPU] Computed layers " << layer_start_ << "-" << layer_end_ << std::endl;
        return true;
    }

    bool Unload() override {
        std::cout << "[CPU] Unloading layers..." << std::endl;
        return true;
    }

    std::string GetName() const override {
        return "cpu";
    }

    int GetMemoryUsage() const override {
        return 0;  // Mock: no memory used
    }

private:
    int layer_start_ = 0;
    int layer_end_ = 0;
};

// BackendFactory creates the appropriate backend for the platform
class BackendFactory {
public:
    static std::unique_ptr<InferenceBackend> Create(const std::string& backend_type) {
        if (backend_type == "cpu") {
            return std::make_unique<CPUBackend>();
        }
        // Default to CPU
        return std::make_unique<CPUBackend>();
    }
};

// InferenceServiceImpl implements the gRPC inference service
class InferenceServiceImpl final : public InferenceService::Service {
public:
    InferenceServiceImpl() {
        backend_ = BackendFactory::Create("cpu");
        backend_->Initialize();
    }

    Status ComputeLayer(ServerContext* context, const LayerComputeRequest* request,
                        LayerComputeResponse* response) override {
        std::cout << "[gRPC] ComputeLayer request: model=" << request->model_id()
                  << " layers=" << request->layer_start() << "-" << request->layer_end() << std::endl;

        // Load layers
        backend_->LoadLayers(request->layer_start(), request->layer_end());

        // Forward pass (mock: return input unchanged)
        Tensor output;
        backend_->Forward(request->input(), &output);

        // Set response
        *response->mutable_output() = output;
        response->set_computed_layers(request->layer_end() - request->layer_start() + 1);
        response->set_compute_time_ms(5.0);  // Mock: 5ms compute time

        std::cout << "[gRPC] ComputeLayer complete" << std::endl;
        return Status::OK;
    }

    Status Generate(ServerContext* context, const InferenceRequest* request,
                    InferenceResponse* response) override {
        std::cout << "[gRPC] Generate request: model=" << request->model_id()
                  << " prompt=\"" << request->prompt() << "\"" << std::endl;

        // Mock generation
        response->set_generated_text("Hello! This is a mock response from DiCAI.");
        response->set_tokens_generated(12);
        response->set_total_latency_ms(150.0);
        response->add_provider_path("provider-1");

        return Status::OK;
    }

private:
    std::unique_ptr<InferenceBackend> backend_;
};

}  // namespace dicai

int main(int argc, char** argv) {
    std::string server_address("0.0.0.0:50051");
    if (argc > 1) {
        server_address = argv[1];
    }

    dicai::InferenceServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[DiCAI] Inference server listening on " << server_address << std::endl;

    server->Wait();

    return 0;
}
