#pragma once

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <libgen.h>
#include <chrono>
#include <unistd.h>
#include "tinyjson.hpp"

namespace psdonnx{

class JsonDataset{
public:
    std::string path_;
    std::string json_file_path_;
    std::ofstream json_ofs_;
    std::ifstream json_ifs_;

public:
    /* must set the path, where to save images and json file */
    JsonDataset(const std::string& path){
        path_ = path;
        json_file_path_ = path_ + "/dataset.json";
    }

    ~JsonDataset(){
        close_writer();
        close_reader();
    }

    void init_writer(){
        json_ofs_.open(json_file_path_.c_str(), std::ofstream::out);
    }

    void init_reader(){
        json_ifs_.open(json_file_path_.c_str(), std::ifstream::in);
    }

    void close_writer(){
        if(json_ofs_.is_open()){
            json_ofs_.close();
        }
    }

    void close_reader(){
        if(json_ifs_.is_open()){
            json_ifs_.close();
        }
    }

    bool feed(double timestamp, const cv::Mat& img_front, const cv::Mat& img_left, const cv::Mat& img_right, const cv::Mat& img_rear){
        const std::string tstr = std::to_string(timestamp);

        const std::string img_front_path = path_ + "/" + tstr + "_front.png";
        cv::imwrite(img_front_path.c_str(), img_front);

        const std::string img_left_path = path_ + "/" + tstr + "_left.png";
        cv::imwrite(img_left_path.c_str(), img_left);

        const std::string img_rear_path = path_ + "/" + tstr + "_rear.png";
        cv::imwrite(img_rear_path.c_str(), img_rear);

        const std::string img_right_path = path_ + "/" + tstr + "_right.png";
        cv::imwrite(img_right_path.c_str(), img_right);

        tiny::TinyJson obj;
        obj["timestamp"].Set(timestamp);
        obj["img_front_path"].Set(img_front_path);
        obj["img_left_path"].Set(img_left_path);
        obj["img_rear_path"].Set(img_rear_path);
        obj["img_right_path"].Set(img_right_path);
        std::string str = obj.WriteJson();
        json_ofs_ << str << std::endl;
        fprintf(stderr, "json string: %s\n", str.c_str());

        return true;
    }

    bool load(double& timestamp, cv::Mat& img_front, cv::Mat& img_left, cv::Mat& img_right, cv::Mat& img_rear){
        std::string line;
        if(std::getline(json_ifs_, line)){
            /* parse json */
            tiny::TinyJson obj;
            obj.ReadJson(line);
            timestamp = obj.Get<double>("timestamp");
            std::string img_front_path = obj.Get<std::string>("img_front_path");
            std::string img_left_path = obj.Get<std::string>("img_left_path");
            std::string img_right_path = obj.Get<std::string>("img_right_path");
            std::string img_rear_path = obj.Get<std::string>("img_rear_path");
            fprintf(stderr, "timestamp: %f, img_front_path: %s, img_left_path: %s, img_right_path: %s, img_rear_path: %s\n", \
                    timestamp, img_front_path.c_str(), img_left_path.c_str(), img_right_path.c_str(), img_rear_path.c_str());
            
            img_front = cv::imread(img_front_path, cv::IMREAD_COLOR);
            img_left = cv::imread(img_left_path, cv::IMREAD_COLOR);
            img_right = cv::imread(img_right_path, cv::IMREAD_COLOR);
            img_rear = cv::imread(img_rear_path, cv::IMREAD_COLOR);

            return true;
        }
        return false;
    }

    void test_writer(){
        init_writer();
        cv::Mat img = cv::Mat::ones(1080, 1920, CV_8UC3);
        double timestamp = 0.01f;
        while(timestamp < 10.0f){
            feed(timestamp, img, img, img, img);
            timestamp += 1.0f;
        }
        close_writer();
    }

    void test_reader(){
        init_reader();
        double timestamp = -1.0f;
        cv::Mat img_front, img_left, img_rear, img_right;
        while(load(timestamp, img_front, img_left, img_right, img_rear)){
            fprintf(stderr, "load dataset t: %f, h: %d, w: %d\n", timestamp, img_front.rows, img_rear.cols);
        }
        close_reader();
    }

};

}