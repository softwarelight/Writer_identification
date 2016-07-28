#ifndef _FISHER_MODEL_H
#define _FISHER_MODEL_H

#include <opencv2/opencv.hpp>
#include "gmm.h"
#include "retrival.h"

class FisherModel
{
public:
    FisherModel();

    FisherModel(int num_comp);

    int get_features(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst);

    int load_features(std::string src);

    int computeCodebook(float use_pca, std::string pca_dst);

    bool saveCodebook(const std::string codebookFile);

    bool loadCodebook(const std::string codebookFile);

    void nomalization(float *data, int size) const;

    VlGMM* get_codebook();
    ~FisherModel();

public:
    int _numComponents;
    int _FeatureDim;

    int _patch_num;
    float* _descriptors;
    VlGMM* _gmm;

    float* _means ;       //_FeatureDim * _numComponents
    float* _covariances ; //_FeatureDim * _numComponents
    float* _priors ;      //_numComponents * 1
};

#endif
