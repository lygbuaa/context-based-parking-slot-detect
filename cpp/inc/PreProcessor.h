#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>

namespace psdonnx
{

class PreProcessor
{
public:
    PreProcessor(){}
    ~PreProcessor(){}

public:
    static std::string gen_output_path(const std::string input_img_path){
        const char* file_name = basename(const_cast<char*>(input_img_path.c_str()));
        std::string output_path = "./output/";
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