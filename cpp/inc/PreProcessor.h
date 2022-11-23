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

    static void test(){
        cv::Mat img = cv::Mat::ones(1280, 1280, CV_8UC3);
        int h = img.rows;
        int w = img.cols;
        // calc roi_w:roi_h = 1:3
        int roi_w = h / 3;
        int roi_h = roi_w * 3;
        cv::Rect roi(0, 0, roi_w, roi_h);
        cv::Mat croped_img = crop(img, roi);
        cv::Mat resized_img = resize(img, 640, 640);
        cv::Mat norm_img = normalize(croped_img);
        mat_2_vec(norm_img);
        cv::Mat stand_img = standardize(resized_img);
        mat_2_vec(stand_img);
        fprintf(stderr, "original_img w: %d, h: %d, type: %d, first: %d\n", img.cols, img.rows, img.type(), img.at<cv::Vec3b>(0, 0)[0]);
        fprintf(stderr, "croped_img w: %d, h: %d, type: %d, first: %d\n", croped_img.cols, croped_img.rows, croped_img.type(), croped_img.at<cv::Vec3b>(0, 0)[0]);
        fprintf(stderr, "resized_img img w: %d, h: %d, type: %d, first: %d\n", resized_img.cols, resized_img.rows, resized_img.type(), resized_img.at<cv::Vec3b>(0, 0)[0]);
        fprintf(stderr, "norm_img w: %d, h: %d, type: %d, first: %f\n", norm_img.cols, norm_img.rows, norm_img.type(), norm_img.at<cv::Vec3f>(0, 0)[0]);
        fprintf(stderr, "stand_img w: %d, h: %d, type: %d, first: %f\n", stand_img.cols, stand_img.rows, stand_img.type(), stand_img.at<cv::Vec3f>(0, 0)[0]);
    }

    static cv::Mat crop(const cv::Mat& img, const cv::Rect& roi){
        // specifies the region of interest in Rectangle form
        return img(roi).clone(); 
    }

    static cv::Mat resize(const cv::Mat& img, const int out_w, const int out_h){
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        return resized_img;
    }

    static cv::Mat normalize(const cv::Mat& img){
        cv::Mat norm_img;
        img.convertTo(norm_img, CV_32FC3, 1.0f/255, 0.0f);  //divided by 255
        return norm_img;
    }

    static cv::Mat standardize(const cv::Mat& img){
        static const float bgr_mean[3] = {131.598f, 129.137f, 131.301f};
        static const float bgr_std[3] = {59.811f, 58.877f, 59.523f};
        
	    std::vector<cv::Mat> bgr_ch(3);
        cv::split(img, bgr_ch);
        //blue chanel
        for(int i=0; i<3; i++){
            bgr_ch[i].convertTo(bgr_ch[i], CV_32F, 1.0f, -1*bgr_mean[i]);
            bgr_ch[i].convertTo(bgr_ch[i], CV_32F, 1.0f/bgr_std[i], 0.0f);
        }

        cv::Mat stand_img;
        cv::merge(bgr_ch, stand_img);
        return stand_img;
    }

    static std::vector<float> mat_2_vec(const cv::Mat& img){
        std::vector<float> array;
        bool is_conti = img.isContinuous();
        if (is_conti) {
        // array.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
            array.assign(img.data, img.data + img.total() * img.channels());
        } else {
            for (int i = 0; i < img.rows; ++i) {
                array.insert(array.end(), img.ptr<float>(i), img.ptr<float>(i)+img.cols*img.channels());
            }
        }
        fprintf(stderr, "img is_conti: %d, img.total: %ld, img.channels: %d, array size: %ld\n", is_conti, img.total(), img.channels(), array.size());
        return array;
    }

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