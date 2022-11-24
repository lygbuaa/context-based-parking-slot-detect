#include <iostream>
#include <fstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "TestCarla.h"


void signal_handler(int sig_num){
	std::cout << "\n@q@ --> it's quit signal: " << sig_num << ", see you later.\n";
	exit(sig_num);
}

bool is_cuda_avaliable(){
    std::cout << "opencv version: " << CV_VERSION << std::endl;
    int cnt = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "getCudaEnabledDeviceCount: " << cnt << std::endl;
    for(int i=0; i<cnt; ++i){
        cv::cuda::printCudaDeviceInfo(i);
    }
    if(cnt > 0){
        std::cout << "current cuda device: " << cv::cuda::getDevice() << std::endl;
    }
    return (cnt > 0);
}

int main(int argc, char* argv[]){
    fprintf(stderr, "\n@i@ --> KittiFlow launched.\n");
    for(int i = 0; i < argc; i++){
        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    }

    is_cuda_avaliable();

    const std::string base_dir = "/mnt/c/work/github/context-based-parking-slot-detect/";
    const std::string dataset_dir = base_dir + "pil_park/carla_town04/image";
    const std::string pcr_model_path = base_dir + "export/pcr.onnx";
    const std::string psd_model_path = base_dir + "export/psd.nms.onnx";
    const std::string json_file_path = base_dir + "result/results.json";
    // psdonnx::OnnxWrapper onnx_wrapper;
    // psdonnx::PreProcessor::test();
    // onnx_wrapper.load_pcr_model(pcr_model_path);
    // onnx_wrapper.test_pcr_model();
    // onnx_wrapper.load_psd_model(psd_model_path);
    // onnx_wrapper.test_psd_model();
    psdonnx::TestCarla test_carla(base_dir, json_file_path);
    const std::deque<std::string> img_path_list = test_carla.list_dir(dataset_dir);
    test_carla.run_test();

    return 0;
}
