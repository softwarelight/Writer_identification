#ifndef _ENCODING_H
#define _ENCODING_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <stdint.h>

#include "recog.hpp"

class Encoding
{
public:
    Encoding(hfrecog& recog, cv::Size img_size);
    virtual int extract_feature(const cv::Mat& image, std::vector<float>& fea, int step) = 0;
    virtual ~Encoding();

private:  
    hfrecog& _recog;
    cv::Size _sz;   

//protected:
public:
    float* _feature;
    int _capacity;
    int _ftrDims;  
    int _ftrNums;

    int _extract_feature_cnn(const cv::Mat& image, int step, int& fea_num, std::vector<cv::Point>& point);
    int64_t _mat_sum(const cv::Mat& image) const;
    void nomalization(float *data, int size) const;
};

#endif