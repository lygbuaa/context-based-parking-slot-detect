#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "tinyjson.hpp"
#include "PreProcessor.h"
#include "OnnxWrapper.h"

namespace psdonnx
{
class TestCarla{
public:
    std::string base_dir_;
    std::string json_file_path_;
    std::string pcr_model_path_;
    std::string psd_model_path_;
    std::unique_ptr<OnnxWrapper> onnx_wrapper_;

public:
    TestCarla(std::string base_dir, std::string json_file_path){
        base_dir_ = base_dir;
        json_file_path_ = json_file_path;
        pcr_model_path_ = base_dir + "export/pcr.onnx";
        psd_model_path_ = base_dir + "export/psd.nms.onnx";
        onnx_wrapper_ = std::unique_ptr<OnnxWrapper>(new OnnxWrapper());
    }
    ~TestCarla(){}

    void run_test(){
        load_model();
        load_json();
    }

    bool load_json(){
        std::ifstream ifs(json_file_path_, std::ifstream::in);
        // std::stringstream buffer;
        if(!ifs.is_open()){
            fprintf(stderr, "open json file failed: %s\n", json_file_path_.c_str());
            return false;
        }

        std::string line;
        while(std::getline(ifs, line)){
            // std::cout << line << std::endl;
            Detections_t det;
            /* parse json */
            tiny::TinyJson obj;
            obj.ReadJson(line);
            det.idx = obj.Get<int>("idx");
            det.img_path = base_dir_ + obj.Get<std::string>("img_path");
            det.angle = obj.Get<int>("angle");
            std::cout << "idx-" << det.idx << ", angle: " << det.angle << ", img: " << det.img_path << std::endl;

            cv::Mat img = cv::imread(det.img_path, cv::IMREAD_COLOR);
            PreProcessor::bgr2rgb(img);
            cv::Vec3b pixel = img.at<cv::Vec3b>(0, 0);
            det.h = img.rows;
            det.w = img.cols;
            printf("original_img w: %d, h: %d, type: %d, first pixel: %d-%d-%d\n", det.w, det.h, img.type(), pixel[0], pixel[1], pixel[2]);
            onnx_wrapper_ -> run_pcr_model(img, det);
            onnx_wrapper_ -> run_psd_model(img, det.angle, det);

            draw_parklots(img, det);
        }

        ifs.close();
        return true;
    }

    void draw_parklots(cv::Mat& img, Detections_t& det){
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

        std::string save_path = gen_output_path(det.img_path);
        cv::imwrite(save_path, img);
    }

    void load_model(){
        onnx_wrapper_ -> load_pcr_model(pcr_model_path_);
        onnx_wrapper_ -> load_psd_model(psd_model_path_);
    }

    std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = base_dir_ + "/result/";
        output_path += file_name;
        output_path += ".psd.png";
        return output_path;
    }

    static uint64_t current_micros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

    static std::deque<std::string> list_dir(const std::string dirpath){
        DIR* dp;
        std::deque<std::string> v_file_list;
        dp = opendir(dirpath.c_str());
        if (nullptr == dp){
            std::cout << "read dirpath failed: " << dirpath << std::endl;
            return v_file_list;
        }

        struct dirent* entry;
        while((entry = readdir(dp))){
            if(DT_DIR == entry->d_type){
                std::cout << "subdirectory ignored: " << entry->d_name << std::endl;
                continue;
            }else if(DT_REG == entry->d_type){
                std::string filepath = dirpath + "/" + entry->d_name;
                v_file_list.emplace_back(filepath);
            }
        }
        //sort into ascending order
        std::sort(v_file_list.begin(), v_file_list.end());
        // for(auto& fp : v_file_list){
        //     LOG(INFO) << "filepath: " << fp;
        // }

        return v_file_list;
    }

};
}