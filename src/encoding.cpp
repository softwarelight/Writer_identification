#include "encoding.h"
#include <fstream>
Encoding::Encoding(hfrecog& recog, cv::Size img_size): _recog(recog), _sz(img_size), _feature(NULL)
{
    _ftrDims = NULL;
    _capacity = 10000;
}

Encoding::~Encoding()
{
    if(_feature != NULL) {
        delete []_feature;
        _feature = NULL;
    }
}

int Encoding::_extract_feature_cnn(const cv::Mat& img, int step, int& fea_num, std::vector<cv::Point>& position)
{
    int width = _sz.width;
    int height = _sz.height;
    float ratio = 0.94; //0.92
    double thresh = (double)_sz.area()*255 * ratio;
    vector<float> feature;

    //std::ofstream log("log_feature_int.txt", std::ios::out);

    float img_row = img.rows;
    float img_col = img.cols;
    int nums = 0;
    for(int r = 0; r <= img_row - height; r += step) {
        if(r> img_row*0.75) {
            if(nums<1000) { //800
                ratio += 0.3; //0.2
                ratio = std::min(ratio,(float)0.99); //0.97
                thresh = (double)_sz.area()*255 * ratio;
            } else if (nums>7000) { //5000
                ratio -=0.2;
                ratio = std::max(ratio,(float)0.75);
                thresh = (double)_sz.area()*255 * ratio;
            }
        }
        else if(r> img_row/2) {
            if(nums<1000) { //800
                ratio += 0.3; //0.2
                ratio = std::min(ratio,(float)0.99);//0.97
                thresh = (double)_sz.area()*255 * ratio;
            } else if (nums>4000) { //3000
                ratio -=0.1; //0.2
                ratio = std::max(ratio,(float)0.75);
                thresh = (double)_sz.area()*255 * ratio;
            }
        }
        else if(r > img_row/3) {
            if(nums<600) { //600
                ratio +=0.3; //0.2
                ratio = std::min(ratio,(float)0.99); //0.96
                thresh = (double)_sz.area()*255 * ratio;
            } else if (nums>2000) { //1500
                ratio -=0.1; //0.2
                ratio = std::max(ratio,(float)0.75);
                thresh = (double)_sz.area()*255 * ratio;
            }
        } 
        for(int c = 0; c <= img_col - width; c += step) {            
            cv::Rect rect(c, r, width, height);
            cv::Mat img_roi = img(rect);
            if(_mat_sum(img_roi) < thresh) {
                feature.clear();                
                _recog.imrecog(img_roi,feature); 
                if(_feature == NULL) {
                    _ftrDims = feature.size();
                    _feature = (float*)malloc(sizeof(float)*_capacity*_ftrDims);
                } else if(_capacity < nums) {
                    float* temp = (float*)malloc(sizeof(float)*_capacity*_ftrDims*2);
                    memcpy(temp, _feature, sizeof(float)*_capacity*_ftrDims); 
                    delete []_feature;
                    _feature = temp;
                    _capacity *=2;
                }

                //nomalization(&feature[0], feature.size());
                //fea.push_back(feature);
               /* for(int k = 0 ; k<feature.size(); k++)
                    log<<feature[k]<<std::endl;*/

                memcpy(_feature + nums*_ftrDims, &feature[0], _ftrDims * sizeof(float));
                position.push_back(cv::Point(c+ width/2, r + height/2));
                nums++;
            }
        }
    }
    fea_num = nums;
    return 0;
}


int64_t Encoding::_mat_sum(const cv::Mat& mat) const
{
    int64_t s = 0;
    int channel = mat.channels();
    for(int row=0; row<mat.rows; row++ ) {
        const uchar* ptr = mat.ptr<uchar>(row);//获取第row行的首地址
        for (int col=0; col<mat.cols; col++ ) {
            s += *ptr;
            ptr += channel;
        }
    }
    return s;
}

void Encoding::nomalization(float *data, int size) const
{
    double norm = 0.0f;
    for (size_t i = 0; i < size; ++i)
        norm += data[i] * data[i];
    norm = sqrtf(norm);

    for (size_t i = 0; i < size; ++i)
        data[i] /= norm;
}
