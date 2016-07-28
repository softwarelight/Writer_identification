#ifndef _FISHER_VECTOR
#define _FISHER_VECTOR

#include "gmm.h"
#include "fisher.h"
#include "encoding.h"
#include "fisher_model.h"

class FisherVector:public Encoding{
public:
    FisherVector(hfrecog& recog, cv::Size size ):Encoding(recog, size), _use_pca(0),_pca(NULL){}
    virtual ~FisherVector();
    virtual int extract_feature(const cv::Mat& image, std::vector<float>& fea, int step);

public:    
    FisherModel model;
    cv::PCA *_pca;
    bool _use_pca;
};

#endif