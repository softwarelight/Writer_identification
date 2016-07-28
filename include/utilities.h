#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


#include "retrival.h"
#include "CvxText.h"
#include "FsHelpers.h"
#include "recog.hpp"

// 生成训练metric的数据 ,pca_ratio 设为0表示不用pca, 否则表示降为率（常用0.93）
int save_feature_trian_metic(HfrzRetrival& retrival, const std::string src, const std::string dst, \
        const int test_size, const float pca_ratio);

// 生成训练metric的数据, 但是对于每一个类只取两张图，并且不生成负pair
int save_feature_simple(const std::string src, const std::string dst, const std::string model_path, const float pca_ratio);

// 对训练数据做pca, 并且生成pca 的参数在测试的时候用
int mat_pca(const cv::Mat &indata, const std::string &dst, const float pca_ratio, cv::Mat &outdata);

// 保存结果图, 将反馈的结果拼成一张大图
void save_result_image(cv::Mat img, const Imdb &image_db, std::vector<std::pair<int,float> > &top_result, 
    cv::Size image_size, int test_index, std::string result_dir, int img_class);

// 计算 CMC 统计结果
void CMC_result(HfrzRetrival &retrival, std::string src, int rank, vector<float> &cmc,\
        std::string result_dir, int max_size, bool use_pca);

void CMC_result_two(HfrzRetrival& retrival, std::string gallery_dir, std::string probe_dir, int rank, vector<float> &cmc,\
    std::string result_dir, std::string method, int max_size, bool use_pca);

void CMC_result_sub_dir(HfrzRetrival& retrival, std::string src, vector<float> &cmc, std::string result_dir);

void save_result_image_two(cv::Mat img,vector<cv::Mat> & imdb, vector<int> imdb_class, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class);
#endif
