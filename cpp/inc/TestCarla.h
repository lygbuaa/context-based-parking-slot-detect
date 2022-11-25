#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include "tinyjson.hpp"
#include "PreProcessor.h"
#include "PsdWrapper.h"

namespace psdonnx
{
class TestCarla{
public:
    std::string base_dir_;
    std::string json_file_path_;
    std::string pcr_model_path_;
    std::string psd_model_path_;
    std::unique_ptr<PsdWrapper> psd_wrapper_;

public:
    TestCarla(std::string base_dir, std::string json_file_path){
        base_dir_ = base_dir;
        json_file_path_ = json_file_path;
        pcr_model_path_ = base_dir + "export/pcr.onnx";
        psd_model_path_ = base_dir + "export/psd.nms.onnx";
        psd_wrapper_ = std::unique_ptr<PsdWrapper>(new PsdWrapper());
    }
    ~TestCarla(){}

    void run_test(){
        psd_wrapper_ -> load_model(pcr_model_path_, psd_model_path_);
        load_json_loop();
    }

    bool load_json_loop(){
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
            cv::Vec3b pixel = img.at<cv::Vec3b>(0, 0);
            det.h = img.rows;
            det.w = img.cols;
            printf("original_img w: %d, h: %d, type: %d, first pixel: %d-%d-%d\n", det.w, det.h, img.type(), pixel[0], pixel[1], pixel[2]);
            psd_wrapper_ -> run_model(img, det, true, base_dir_+"/result/");
        }

        ifs.close();
        return true;
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