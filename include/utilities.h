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

// ����ѵ��metric������ ,pca_ratio ��Ϊ0��ʾ����pca, �����ʾ��Ϊ�ʣ�����0.93��
int save_feature_trian_metic(HfrzRetrival& retrival, const std::string src, const std::string dst, \
        const int test_size, const float pca_ratio);

// ����ѵ��metric������, ���Ƕ���ÿһ����ֻȡ����ͼ�����Ҳ����ɸ�pair
int save_feature_simple(const std::string src, const std::string dst, const std::string model_path, const float pca_ratio);

// ��ѵ��������pca, ��������pca �Ĳ����ڲ��Ե�ʱ����
int mat_pca(const cv::Mat &indata, const std::string &dst, const float pca_ratio, cv::Mat &outdata);

// ������ͼ, �������Ľ��ƴ��һ�Ŵ�ͼ
void save_result_image(cv::Mat img, const Imdb &image_db, std::vector<std::pair<int,float> > &top_result, 
    cv::Size image_size, int test_index, std::string result_dir, int img_class);

// ���� CMC ͳ�ƽ��
void CMC_result(HfrzRetrival &retrival, std::string src, int rank, vector<float> &cmc,\
        std::string result_dir, int max_size, bool use_pca);

void CMC_result_two(HfrzRetrival& retrival, std::string gallery_dir, std::string probe_dir, int rank, vector<float> &cmc,\
    std::string result_dir, std::string method, int max_size, bool use_pca);

void CMC_result_sub_dir(HfrzRetrival& retrival, std::string src, vector<float> &cmc, std::string result_dir);

void save_result_image_two(cv::Mat img,vector<cv::Mat> & imdb, vector<int> imdb_class, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class);
#endif
