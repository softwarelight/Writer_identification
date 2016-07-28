#ifndef __CODEBOOK_H__
#define __CODEBOOK_H__

#include <fstream>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
#include "retrival.h"

class Codebook
{
public:
    Codebook();

    int get_features(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst);

    int load_features(std::string src);

    int computeCodebook();

    bool saveCodebook(const std::string codebookFile);

    bool loadCodebook(const std::string codebookFile, VlKMeans** pc);

    bool codeCodebook(float *descriptors, int descriptorCount, vl_uint32 *puiAssignments, float *norm);

    VlKMeans* get_codebook();
    ~Codebook();

private:
    int _codebookSize;
    int _FeatureDim;

    int _patch_num;
    float* _descriptors;
    VlKMeans *_pCodebook;
};

#endif
