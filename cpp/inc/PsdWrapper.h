#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "tinyjson.hpp"
#include "PreProcessor.h"
#include "OnnxWrapper.h"

namespace psdonnx{
class PsdWrapper
{
public:
    std::unique_ptr<OnnxWrapper> onnx_wrapper_;

public:
    PsdWrapper(){
        onnx_wrapper_ = std::unique_ptr<OnnxWrapper>(new OnnxWrapper());
    }
    ~PsdWrapper(){}

    bool load_model(const std::string& pcr_path, const std::string& psd_path){
        onnx_wrapper_ -> load_pcr_model(pcr_path);
        onnx_wrapper_ -> load_psd_model(psd_path);
        return true;
    }

    bool run_model(cv::Mat& img, Detections_t& det, bool debug_save=false, const std::string& path="./"){
        PreProcessor::bgr2rgb(img);
        onnx_wrapper_ -> run_pcr_model(img, det);
        onnx_wrapper_ -> run_psd_model(img, det.angle, det);
        if(debug_save){
            draw_parklots(img, det, path);
        }
        return true;
    }

    void draw_parklots(cv::Mat& img, Detections_t& det, const std::string& path="./"){
        char tmp[32] = {0};
        sprintf(tmp, "%d: %d, %d", det.idx, det.type, int(det.angle));
        cv::Point org(det.w-120, 20);
        cv::Scalar yellow(0, 255, 255);
        cv::putText(img, tmp, org, cv::FONT_HERSHEY_DUPLEX, 0.6, yellow, 1);
        
        for(Parklot_t parklot : det.parklots){
            cv::Scalar line_color(0, 0, 255);
            bool is_empty = (parklot.label == 0);
            if(is_empty){
                line_color = cv::Scalar(0, 255, 0);
            }
            std::vector<cv::Point> vertices;
            for(int i=0; i<4; i++){
                vertices.emplace_back(parklot.quads[2*i+0], parklot.quads[2*i+1]);
            }
            for(int i=0; i<4; i++){
                cv::line(img, vertices[i], vertices[(i+1)%4], line_color, 2, cv::LINE_8);
            }
            memset(tmp, 0, 32);
            sprintf(tmp, "%.2f", parklot.score);
            int xmin = int(parklot.bbox[0]);
            if(xmin < 0){
                xmin = 0;
            }
            int ymin = int(parklot.bbox[1]);
            if(ymin < 0){
                ymin = 0;
            }
            cv::Point org(xmin, ymin);
            cv::putText(img, tmp, org, cv::FONT_HERSHEY_DUPLEX, 0.6, yellow, 1);
        }

        std::string save_path = path + gen_save_name(det.img_path);
        cv::imwrite(save_path, img);
    }

    std::string gen_save_name(const std::string input_img_path){
        std::string save_name = basename(const_cast<char*>(input_img_path.c_str()));
        save_name += ".psd.png";
        return save_name;
    }

    static uint64_t current_micros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

};
}