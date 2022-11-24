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
        cv::Vec3b original_pixel = img.at<cv::Vec3b>(0, 0);
        fprintf(stderr, "original_img w: %d, h: %d, type: %d, first-pixel: %d-%d-%d\n", img.cols, img.rows, img.type(), original_pixel[0], original_pixel[1], original_pixel[2]);
        cv::Vec3b croped_pixel = croped_img.at<cv::Vec3b>(0, 0);
        fprintf(stderr, "croped_img w: %d, h: %d, type: %d, first-pixel: %d-%d-%d\n", croped_img.cols, croped_img.rows, croped_img.type(), croped_pixel[0], croped_pixel[1], croped_pixel[2]);
        cv::Vec3b resized_pixel = resized_img.at<cv::Vec3b>(0, 0);
        fprintf(stderr, "resized_img img w: %d, h: %d, type: %d, first-pixel: %d-%d-%d\n", resized_img.cols, resized_img.rows, resized_img.type(), resized_pixel[0], resized_pixel[1], resized_pixel[2]);
        cv::Vec3f norm_pixel = norm_img.at<cv::Vec3f>(0, 0);
        fprintf(stderr, "norm_img w: %d, h: %d, type: %d, first-pixel: %f-%f-%f\n", norm_img.cols, norm_img.rows, norm_img.type(), norm_pixel[0], norm_pixel[1], norm_pixel[2]);
        cv::Vec3f stand_pixel = stand_img.at<cv::Vec3f>(0, 0);
        fprintf(stderr, "stand_img w: %d, h: %d, type: %d, first-pixel: %f-%f-%f\n", stand_img.cols, stand_img.rows, stand_img.type(), stand_pixel[0], stand_pixel[1], stand_pixel[2]);
    }

    static void bgr2rgb(cv::Mat& img){
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
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
        cv::Vec3f norm_pixel = norm_img.at<cv::Vec3f>(0, 0);
        fprintf(stderr, "norm_img w: %d, h: %d, type: %d, first-pixel: %f-%f-%f\n", norm_img.cols, norm_img.rows, norm_img.type(), norm_pixel[0], norm_pixel[1], norm_pixel[2]);
        return norm_img;
    }

    /* input img should be rgb format */
    static cv::Mat standardize(const cv::Mat& img){
        static const float rgb_mean[3] = {131.301f, 129.137f, 131.598f};
        static const float rgb_std[3] = {59.523f, 58.877f, 59.811f};
        
	    std::vector<cv::Mat> rgb_ch(3);
        cv::split(img, rgb_ch);
        //blue chanel
        for(int i=0; i<3; i++){
            rgb_ch[i].convertTo(rgb_ch[i], CV_32F, 1.0f, -1*rgb_mean[i]);
            rgb_ch[i].convertTo(rgb_ch[i], CV_32F, 1.0f/rgb_std[i], 0.0f);
        }

        cv::Mat stand_img;
        cv::merge(rgb_ch, stand_img);
        return stand_img;
    }

    static std::vector<float> mat_2_vec(const cv::Mat& img){
        std::vector<float> array;
        bool is_conti = img.isContinuous();
        if (is_conti) {
            /* should not use img.data */
            array.assign(img.ptr<float>(0), img.ptr<float>(0) + img.total() * img.channels());
        } else {
            for (int i = 0; i < img.rows; ++i) {
                array.insert(array.end(), img.ptr<float>(i), img.ptr<float>(i)+img.cols*img.channels());
            }
        }
        // cv::Vec3f img_pixel = img.at<cv::Vec3f>(0, 0);
        // fprintf(stderr, "img is_conti: %d, img.total: %ld, img.channels: %d, array size: %ld\n", is_conti, img.total(), img.channels(), array.size());
        // fprintf(stderr, "img first-pixel: %f-%f-%f\n", img_pixel[0], img_pixel[1], img_pixel[2]);
        // fprintf(stderr, "array first-pixel: %f-%f-%f\n", array[0], array[1], array[2]);
        return array;
    }

};

}