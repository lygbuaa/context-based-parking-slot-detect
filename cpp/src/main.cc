#include <iostream>
#include <fstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "OnnxWrapper.h"


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

    const std::string dataset_dir = "/home/hugoliu/github/context-based-parking-slot-detect/pil_park/carla_town04/image";
    const std::string pcr_model_path = "/home/hugoliu/github/context-based-parking-slot-detect/export/pcr.onnx";
    const std::string psd_model_path = "/home/hugoliu/github/context-based-parking-slot-detect/export/psd.nms.onnx";
    psdonnx::OnnxWrapper onnx_wrapper;
    // const std::deque<std::string> img_path_list = onnx_wrapper.list_dir(dataset_dir);
    onnx_wrapper.load_pcr_model(pcr_model_path);
    // onnx_wrapper.run_pcr(img_path_list);

    return 0;
}
