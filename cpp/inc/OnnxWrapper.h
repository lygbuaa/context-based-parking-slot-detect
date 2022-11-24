#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "PreProcessor.h"

namespace psdonnx
{
typedef struct{
    // x0, y0, x1, y1
    float bbox[4] = {-1.0f};
    int64_t label = -1;
    float quads[8] = {-1.0f};
    float score = -1.0f;
}Parklot_t;

typedef struct{
    int idx = -1;
    std::string img_path;
    float angle = 0.0f;
    int type = -1;
    int h = 0;
    int w = 0;
    std::vector<Parklot_t> parklots;
}Detections_t;

class OnnxWrapper
{
public:
/* replaced by CheckStatus() */
    #define ORT_ABORT_ON_ERROR(expr)                             \
    do {                                                       \
        OrtStatus* onnx_status = (expr);                         \
        if (onnx_status != NULL) {                               \
            const char* msg = g_ort_->GetErrorMessage(onnx_status); \
            fprintf(stderr, "%s\n", msg);                          \
            g_ort_->ReleaseStatus(onnx_status);                    \
            abort();                                               \
        }                                                        \
    } while (0);

    typedef struct{
        OrtSession* sess = nullptr;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_dims;
        std::vector<ONNXTensorElementDataType> input_types;
        std::vector<OrtValue*> input_tensors;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_dims;
        std::vector<ONNXTensorElementDataType> output_types;
        std::vector<OrtValue*> output_tensors;
    }ORT_S_t;

private:
    const OrtApi* g_ort_ = nullptr;
    const OrtApiBase* g_ort_base_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;

    ORT_S_t g_pcr_s_;
    ORT_S_t g_psd_s_;
    Detections_t g_det_;

public:
    OnnxWrapper(){
        init_ort();
    }

    ~OnnxWrapper(){
        destroy_ort();
    }

    bool load_pcr_model(const std::string& model_path){
        return load_model(model_path, g_pcr_s_);
    }

    bool load_psd_model(const std::string& model_path){
        return load_model(model_path, g_psd_s_);
    }

    /* img should be rgb format */
    bool run_pcr_model(const cv::Mat& img, Detections_t& det){
        /* define input tensor size, could retrieve from g_pcr_s_, too */
        static constexpr int PCR_W = 64;
        static constexpr int PCR_H = 192;
        /* preprocess */
        const int h = img.rows;
        const int w = img.cols;
        // calc roi_w:roi_h = 1:3
        int roi_w = h / 3;
        int roi_h = roi_w * 3;
        cv::Rect roi(0, 0, roi_w, roi_h);
        cv::Mat croped_img = PreProcessor::crop(img, roi);
        cv::Mat resized_img = PreProcessor::resize(croped_img, PCR_W, PCR_H);
        cv::Mat stand_img = PreProcessor::standardize(resized_img);

        /* prepare input data */
        std::vector<const char*>& input_node_names = g_pcr_s_.input_node_names;
        std::vector<std::vector<int64_t>>& input_node_dims = g_pcr_s_.input_node_dims;
        std::vector<ONNXTensorElementDataType>& input_types = g_pcr_s_.input_types;
        std::vector<OrtValue*>& input_tensors = g_pcr_s_.input_tensors;

        std::vector<const char*>& output_node_names = g_pcr_s_.output_node_names;
        std::vector<std::vector<int64_t>>& output_node_dims = g_pcr_s_.output_node_dims;
        std::vector<ONNXTensorElementDataType>& output_types = g_pcr_s_.output_types;
        std::vector<OrtValue*>& output_tensors = g_pcr_s_.output_tensors;

        size_t input_img_size = 1;
        for(int k=0; k<input_node_dims[0].size(); ++k){
            input_img_size *= input_node_dims[0][k];
            // fprintf(stderr, "%d-input_node_dims[0][k]: %ld, input_img_size: %ld\n", k, input_node_dims[0][k], input_img_size);
        }
        size_t input_img_length = input_img_size * sizeof(float);
        std::vector<float> input_img_fp32 = PreProcessor::mat_2_vec(stand_img);
        // fprintf(stderr, "input_img_length: %ld, input_img_fp32 size: %ld\n", input_img_length, input_img_fp32.size());

        /* only 1 input, move into input_tensors[0] */
        OrtMemoryInfo* memory_info;
        CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                        memory_info, reinterpret_cast<void*>(input_img_fp32.data()), input_img_length,
                        input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
        g_ort_->ReleaseMemoryInfo(memory_info);

        /* do inference */
        CheckStatus(g_ort_->Run(g_pcr_s_.sess, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                        input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data()));

        /* postprocess */
        float angle = 0.0;
        float type_prob_max = -1.0f;
        int type = -1;
        for(int i=0; i<output_node_names.size(); i++){
            void* output_buffer;
            CheckStatus(g_ort_->GetTensorMutableData(output_tensors[i], &output_buffer));
            float* float_buffer = reinterpret_cast<float*>(output_buffer);
            int output_size = 1;
            for(int k=0; k<output_node_dims[i].size(); k++){
                output_size *= output_node_dims[i][k];
            }
            fprintf(stderr, "output[%d] -  %s: \n", i, output_node_names[i]);
            for(int k=0; k<output_size; k++){
                fprintf(stderr, "%f\n", float_buffer[k]);
            }

            /* retrieve angle */
            if(i==0){
                angle = float_buffer[0] * 180.0f - 90.0f;
            }
            /* retrieve type */
            else if(i==1){
                for(int k=0; k<output_size; k++){
                    if(float_buffer[k] > type_prob_max){
                        type_prob_max = float_buffer[k];
                        type = k;
                    }
                }
            }

        }
        fprintf(stderr, "pcr angle: %f\n", angle);
        fprintf(stderr, "pcr type: %d, prob: %f\n", type, type_prob_max);
        det.angle = angle;
        det.type = type;

        return true;
    }

    /* img should be rgb format */
    bool run_psd_model(const cv::Mat& img, const float angle, Detections_t& det){
        /* define input tensor size, could retrieve from g_psd_s_, too */
        static constexpr int PSD_W = 640;
        static constexpr int PSD_H = 640;
        /* preprocess */
        cv::Mat resized_img = PreProcessor::resize(img, PSD_W, PSD_H);
        cv::Mat norm_img = PreProcessor::normalize(resized_img);

        /* prepare input data */
        std::vector<const char*>& input_node_names = g_psd_s_.input_node_names;
        std::vector<std::vector<int64_t>>& input_node_dims = g_psd_s_.input_node_dims;
        std::vector<ONNXTensorElementDataType>& input_types = g_psd_s_.input_types;
        std::vector<OrtValue*>& input_tensors = g_psd_s_.input_tensors;

        std::vector<const char*>& output_node_names = g_psd_s_.output_node_names;
        std::vector<std::vector<int64_t>>& output_node_dims = g_psd_s_.output_node_dims;
        std::vector<ONNXTensorElementDataType>& output_types = g_psd_s_.output_types;
        std::vector<OrtValue*>& output_tensors = g_psd_s_.output_tensors;

        OrtMemoryInfo* memory_info;
        CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        std::vector<float> input_angle_fp32;
        input_angle_fp32.emplace_back(angle);
        size_t input_angle_length = 1 * sizeof(float);

        /* input[0]-angle, type: 1, dims: (1,), move into input_tensors[0] */
        CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                        memory_info, reinterpret_cast<void*>(input_angle_fp32.data()), input_angle_length,
                        input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));

        size_t input_img_size = 1;
        for(int k=0; k<input_node_dims[1].size(); ++k){
            input_img_size *= input_node_dims[1][k];
            // fprintf(stderr, "%d-input_node_dims[0][k]: %ld, input_img_size: %ld\n", k, input_node_dims[0][k], input_img_size);
        }
        size_t input_img_length = input_img_size * sizeof(float);
        std::vector<float> input_img_fp32 = PreProcessor::mat_2_vec(norm_img);
        // fprintf(stderr, "input_img_length: %ld, input_img_fp32 size: %ld\n", input_img_length, input_img_fp32.size());

        /* input[1]-image, type: 1, dims: (1,640,640,3,), move into input_tensors[1] */
        CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                        memory_info, reinterpret_cast<void*>(input_img_fp32.data()), input_img_length,
                        input_node_dims[1].data(), input_node_dims[1].size(), input_types[1], &input_tensors[1]));
        g_ort_->ReleaseMemoryInfo(memory_info);

        /* do inference */
        CheckStatus(g_ort_->Run(g_psd_s_.sess, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                        input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data()));

        /* postprocess */
        int64_t pkl_num = -1;
        std::vector<Parklot_t>& parklots = det.parklots;
        std::vector<float> boxes;
        std::vector<int64_t> labels;
        std::vector<float> quads;
        std::vector<float> scores;
        for(int i=0; i<output_node_names.size(); i++){
            void* output_buffer;
            CheckStatus(g_ort_->GetTensorMutableData(output_tensors[i], &output_buffer));
            float* float_buffer = reinterpret_cast<float*>(output_buffer);
            int64_t* int64_buffer = reinterpret_cast<int64_t*>(output_buffer);

            /* dynamic output shape, we should retrieve again */
            OrtTensorTypeAndShapeInfo* shape_info;
            CheckStatus(g_ort_->CreateTensorTypeAndShapeInfo(&shape_info));
            CheckStatus(g_ort_->GetTensorTypeAndShape(output_tensors[i], &shape_info));
            size_t num_dims;
            CheckStatus(g_ort_->GetDimensionsCount(shape_info, &num_dims));
            std::vector<int64_t> out_node_dims;
            out_node_dims.resize(num_dims);
            CheckStatus(g_ort_->GetDimensions(shape_info, out_node_dims.data(), num_dims));

            int output_size = 1;
            std::string dimstr="(";
            for(int k=0; k<num_dims; ++k){
                output_size *= out_node_dims[k];
                dimstr += std::to_string(out_node_dims[k]);
                dimstr += ",";
            }
            dimstr += ")";

            fprintf(stderr, "output[%d]-%s, type: %d, dims: %s, output_size: %d\n", i, output_node_names[i], output_types[i], dimstr.c_str(), output_size);
            g_ort_->ReleaseTensorTypeAndShapeInfo(shape_info);

            // for(int k=0; k<output_size; k++){
            //     fprintf(stderr, "%f\n", float_buffer[k]);
            // }

            /* retrieve boxes */
            if(i==0){
                pkl_num = out_node_dims[0];
                boxes.assign(float_buffer, float_buffer+output_size);
            }
            /* retrieve labels */
            else if(i==1){
                labels.assign(int64_buffer, int64_buffer+output_size);
            }
            /* retrieve quads */
            else if(i==2){
                quads.assign(float_buffer, float_buffer+output_size);
            }
            /* retrieve scores */
            else if (i==3){
                scores.assign(float_buffer, float_buffer+output_size);
            }

            /* output dynamic shape, release last output_tensors */
            if(output_tensors[i]){
                g_ort_ -> ReleaseValue(output_tensors[i]);
                output_tensors[i] = nullptr;
            }
        }

        for(int i=0; i<pkl_num; i++){
            Parklot_t pkl;
            for(int j=0; j<4; j++){
                pkl.bbox[j] = boxes[4*i+j];
            }
            pkl.label = labels[i];
            for(int j=0; j<8; j++){
                pkl.quads[j] = quads[8*i+j];
            }
            pkl.score = scores[i];
            det.parklots.emplace_back(pkl);
            fprintf(stderr, "parklot[%d], label: %ld, score: %.2f, bbox:[%.2f, %.2f, %.2f, %.2f], quads:[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", \
            i, pkl.label, pkl.score, pkl.bbox[0], pkl.bbox[1], pkl.bbox[2], pkl.bbox[3], pkl.quads[0], pkl.quads[1], pkl.quads[2], pkl.quads[3], pkl.quads[4], pkl.quads[5], pkl.quads[6], pkl.quads[7]);
        }
        return true;
    }

    void test_pcr_model(){
        cv::Mat img = cv::Mat::ones(640, 640, CV_8UC3);
        PreProcessor::bgr2rgb(img);
        run_pcr_model(img, g_det_);
    }

    void test_psd_model(){
        cv::Mat img = cv::Mat::ones(640, 640, CV_8UC3);
        PreProcessor::bgr2rgb(img);
        float angle = 0.0f;
        run_psd_model(img, angle, g_det_);
    }

private:
    bool load_model(const std::string& model_path, ORT_S_t& model_s){
        fprintf(stderr, "load model: %s", model_path.c_str());

        CheckStatus(g_ort_->CreateSession(env_, model_path.c_str(), session_options_, &model_s.sess));

        OrtAllocator* allocator;
        CheckStatus(g_ort_->GetAllocatorWithDefaultOptions(&allocator));
        size_t num_input_nodes;
        CheckStatus(g_ort_->SessionGetInputCount(model_s.sess, &num_input_nodes));

        std::vector<const char*>& input_node_names = model_s.input_node_names;
        std::vector<std::vector<int64_t>>& input_node_dims = model_s.input_node_dims;
        std::vector<ONNXTensorElementDataType>& input_types = model_s.input_types;
        std::vector<OrtValue*>& input_tensors = model_s.input_tensors;

        input_node_names.resize(num_input_nodes);
        input_node_dims.resize(num_input_nodes);
        input_types.resize(num_input_nodes);
        input_tensors.resize(num_input_nodes);

        for (size_t i = 0; i < num_input_nodes; i++) {
            // Get input node names
            char* input_name;
            CheckStatus(g_ort_->SessionGetInputName(model_s.sess, i, allocator, &input_name));
            input_node_names[i] = input_name;

            // Get input node types
            OrtTypeInfo* typeinfo;
            CheckStatus(g_ort_->SessionGetInputTypeInfo(model_s.sess, i, &typeinfo));
            const OrtTensorTypeAndShapeInfo* tensor_info;
            CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
            ONNXTensorElementDataType type;
            CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
            input_types[i] = type;

            // Get input shapes/dims
            size_t num_dims;
            CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
            input_node_dims[i].resize(num_dims);
            CheckStatus(g_ort_->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims));

            size_t tensor_size;
            CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));

            std::string dimstr="(";
            for(int k=0; k<num_dims; ++k){
                dimstr += std::to_string(input_node_dims[i][k]);
                dimstr += ",";
            }
            dimstr += ")";

            /* print input tensor information */
            fprintf(stderr, "input[%ld]-%s, type: %d, dims: %s\n", i, input_name, type, dimstr.c_str());

            if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
        }

        size_t num_output_nodes;
        std::vector<const char*>& output_node_names = model_s.output_node_names;
        std::vector<std::vector<int64_t>>& output_node_dims = model_s.output_node_dims;
        std::vector<ONNXTensorElementDataType>& output_types = model_s.output_types;
        std::vector<OrtValue*>& output_tensors = model_s.output_tensors;

        CheckStatus(g_ort_->SessionGetOutputCount(model_s.sess, &num_output_nodes));
        fprintf(stderr, "num_output_nodes: %ld\n", num_output_nodes);
        output_node_names.resize(num_output_nodes);
        output_node_dims.resize(num_output_nodes);
        output_tensors.resize(num_output_nodes);
        output_types.resize(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            // Get output node names
            char* output_name;
            CheckStatus(g_ort_->SessionGetOutputName(model_s.sess, i, allocator, &output_name));
            output_node_names[i] = output_name;
            // fprintf(stderr, "%ld-output_name: %s\n", i, output_name);

            OrtTypeInfo* typeinfo;
            CheckStatus(g_ort_->SessionGetOutputTypeInfo(model_s.sess, i, &typeinfo));
            const OrtTensorTypeAndShapeInfo* tensor_info;
            CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
            ONNXTensorElementDataType type;
            CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
            output_types[i] = type;
            // fprintf(stderr, "%ld-type: %d\n", i, type);

            // Get output shapes/dims
            size_t num_dims;
            CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
            output_node_dims[i].resize(num_dims);
            CheckStatus(g_ort_->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));
            // fprintf(stderr, "%ld-num_dims: %ld\n", i, num_dims);

            /* when it's variable output, tensor_size could be negative, so tensor_size will overflow */
            // size_t tensor_size;
            // CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));
            // fprintf(stderr, "%ld-tensor_size: %ld\n", i, tensor_size);

            std::string dimstr="(";
            for(int k=0; k<num_dims; ++k){
                dimstr += std::to_string(output_node_dims[i][k]);
                dimstr += ",";
            }
            dimstr += ")";
            /* print output tensor information */
            fprintf(stderr, "output[%ld]-%s, type: %d, dims: %s\n", i, output_name, type, dimstr.c_str());

            if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
        }

        return true;
    }

    bool CheckStatus(OrtStatus* status) {
        if (status != nullptr) {
            const char* msg = g_ort_->GetErrorMessage(status);
            std::cerr << msg << std::endl;
            g_ort_->ReleaseStatus(status);
            throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
        }
        return true;
    }

    bool init_ort(){
        g_ort_base_ = OrtGetApiBase();
        if (!g_ort_base_){
            fprintf(stderr, "Failed to OrtGetApiBase.\n");
            return false;
        }

        std::cout << "ort version: " << g_ort_base_ -> GetVersionString() << std::endl;

        g_ort_ = g_ort_base_->GetApi(ORT_API_VERSION);
        if (!g_ort_) {
            fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
            return false;
        }

        CheckStatus(g_ort_->CreateEnv(ORT_LOGGING_LEVEL_INFO, "psd", &env_));
        if (!env_) {
            fprintf(stderr, "Failed to CreateEnv.\n");
            return false;
        }

        /* use default session option is ok */
        CheckStatus(g_ort_->CreateSessionOptions(&session_options_));
        // CheckStatus(g_ort_->SetIntraOpNumThreads(session_options_, 1));
        // CheckStatus(g_ort_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
        // std::vector<const char*> options_keys = {"runtime", "buffer_type"};
        // std::vector<const char*> options_values = {backend.c_str(), "FLOAT"};  // set to TF8 if use quantized data
        // CheckStatus(g_ort_->SessionOptionsAppendExecutionProvider(session_options_, "SNPE", options_keys.data(), options_values.data(), options_keys.size()));

        return true;
    }

    void destroy_ort(){
        if(session_options_) g_ort_->ReleaseSessionOptions(session_options_);
        if(g_pcr_s_.sess) g_ort_->ReleaseSession(g_pcr_s_.sess);
        if(g_psd_s_.sess) g_ort_->ReleaseSession(g_psd_s_.sess);
        if(env_) g_ort_->ReleaseEnv(env_);
    }

    void verify_input_output_count(OrtSession* sess) {
        size_t count;
        CheckStatus(g_ort_->SessionGetInputCount(sess, &count));
        assert(count == 1);
        CheckStatus(g_ort_->SessionGetOutputCount(sess, &count));
        assert(count == 1);
    }

    int enable_cuda(OrtSessionOptions* session_options) {
        // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
        OrtCUDAProviderOptions o;
        // Here we use memset to initialize every field of the above data struct to zero.
        memset(&o, 0, sizeof(o));
        // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
        // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
        o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        o.gpu_mem_limit = SIZE_MAX;
        OrtStatus* onnx_status = g_ort_->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
        if (onnx_status != NULL) {
            const char* msg = g_ort_->GetErrorMessage(onnx_status);
            fprintf(stderr, "%s\n", msg);
            g_ort_->ReleaseStatus(onnx_status);
            return -1;
        }
        return 0;
    }

};

}