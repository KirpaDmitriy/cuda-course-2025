#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

using namespace nvinfer1;

const int INPUT_W = 640;
const int INPUT_H = 640;
const float CONF_THRESHOLD = 0.5f;

const char* IN_NAME = "images";
const char* OUT_BOXES = "boxes";
const char* OUT_SCORES = "scores";
const char* OUT_LABELS = "labels";

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kDIRTY)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << " at line " << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

__global__ void preprocessKernel(const uint8_t* __restrict__ src, float* __restrict__ dst, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int area = width * height;

    if (idx < area) {
        int srcIdx = idx * 3;
        
        float b = src[srcIdx + 0];
        float g = src[srcIdx + 1];
        float r = src[srcIdx + 2];
        
        dst[idx]          = ((r / 255.0f) - 0.485f) / 0.229f;
        dst[idx + area]   = ((g / 255.0f) - 0.456f) / 0.224f;
        dst[idx + 2*area] = ((b / 255.0f) - 0.406f) / 0.225f;
    }
}

void launchPreprocess(const uint8_t* src, float* dst, int width, int height, cudaStream_t stream) {
    int numPixels = width * height;
    int blockSize = 256;
    int gridSize = (numPixels + blockSize - 1) / blockSize;
    preprocessKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, width, height);
}

std::vector<std::string> loadClasses(const std::string& path) {
    std::vector<std::string> classes;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) if (!line.empty()) classes.push_back(line);
    return classes;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./RetinaNetTRT <engine> <classes> <video> [out]" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    std::string classPath = argv[2];
    std::string videoPath = argv[3];
    std::string outPath = (argc > 4) ? argv[4] : "output.mp4";

    auto classes = loadClasses(classPath);
    
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) return 1;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return 1;

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (!engine) return 1;

    IExecutionContext* context = engine->createExecutionContext();
    
    auto dim_boxes = engine->getTensorShape(OUT_BOXES); 
    int max_dets = (dim_boxes.nbDims > 1) ? dim_boxes.d[1] : 300; 

    std::cout << "Detected max detections: " << max_dets << std::endl;

    size_t in_size_float = 3 * INPUT_W * INPUT_H * sizeof(float);
    size_t in_size_uint8 = 3 * INPUT_W * INPUT_H * sizeof(uint8_t);

    size_t out_boxes_size = max_dets * 4 * sizeof(float);
    size_t out_scores_size = max_dets * sizeof(float);
    size_t out_labels_size = max_dets * sizeof(int64_t);

    uint8_t* d_input_uint8 = nullptr;
    float*   d_input_float = nullptr;
    void* d_boxes = nullptr;
    void* d_scores = nullptr;
    void* d_labels = nullptr;

    CHECK(cudaMalloc(&d_input_uint8, in_size_uint8));
    CHECK(cudaMalloc(&d_input_float, in_size_float));

    CHECK(cudaMalloc(&d_boxes, out_boxes_size));
    CHECK(cudaMalloc(&d_scores, out_scores_size));
    CHECK(cudaMalloc(&d_labels, out_labels_size));

    context->setTensorAddress(IN_NAME, d_input_float);
    context->setTensorAddress(OUT_BOXES, d_boxes);
    context->setTensorAddress(OUT_SCORES, d_scores);
    context->setTensorAddress(OUT_LABELS, d_labels);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    float* h_boxes;
    float* h_scores;
    int64_t* h_labels;

    CHECK(cudaMallocHost((void**)&h_boxes, out_boxes_size));
    CHECK(cudaMallocHost((void**)&h_scores, out_scores_size));
    CHECK(cudaMallocHost((void**)&h_labels, out_labels_size));

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) return 1;

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(outPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    
    cv::Mat frame, resized;
    resized.create(INPUT_H, INPUT_W, CV_8UC3); 

    float scale_x = (float)width / INPUT_W;
    float scale_y = (float)height / INPUT_H;

    std::cout << "Starting inference with CUDA preprocessing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frames = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));
        
        CHECK(cudaMemcpyAsync(d_input_uint8, resized.data, in_size_uint8, cudaMemcpyHostToDevice, stream));
        
        launchPreprocess(d_input_uint8, d_input_float, INPUT_W, INPUT_H, stream);
        
        bool status = context->enqueueV3(stream);
        if (!status) break;
        
        CHECK(cudaMemcpyAsync(h_boxes, d_boxes, out_boxes_size, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(h_scores, d_scores, out_scores_size, cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(h_labels, d_labels, out_labels_size, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        
        for (int i = 0; i < max_dets; i++) {
            float score = h_scores[i];
            if (score < CONF_THRESHOLD) continue;

            int lbl = (int)h_labels[i];
            
            float x1 = h_boxes[i*4 + 0] * scale_x;
            float y1 = h_boxes[i*4 + 1] * scale_y;
            float x2 = h_boxes[i*4 + 2] * scale_x;
            float y2 = h_boxes[i*4 + 3] * scale_y;
            
            x1 = std::max(0.0f, x1); y1 = std::max(0.0f, y1);
            x2 = std::min((float)width, x2); y2 = std::min((float)height, y2);

            cv::rectangle(frame, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 0), 2);
            
            std::string text = (lbl >= 0 && lbl < classes.size()) ? classes[lbl] : std::to_string(lbl);
            text += " " + std::to_string((int)(score * 100)) + "%";

            int baseLine;
            cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            y1 = std::max(y1, (float)labelSize.height);
            cv::rectangle(frame, cv::Point((int)x1, (int)y1 - labelSize.height - 5), 
                                 cv::Point((int)x1 + labelSize.width, (int)y1 + 5), 
                                 cv::Scalar(0, 255, 0), -1);
            cv::putText(frame, text, cv::Point((int)x1, (int)y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        }

        writer.write(frame);
        
        frames++;
        if (frames % 30 == 0) std::cout << "Processed " << frames << " frames" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "Done! Processed " << frames << " frames in " << duration << "s. FPS: " << (duration > 0 ? frames / duration : 0) << std::endl;

    cudaFreeHost(h_boxes);
    cudaFreeHost(h_scores);
    cudaFreeHost(h_labels);
    
    cudaFree(d_input_uint8);
    cudaFree(d_input_float);
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_labels);
    
    delete context;
    delete engine;
    delete runtime;
    
    cudaStreamDestroy(stream);

    return 0;
}
