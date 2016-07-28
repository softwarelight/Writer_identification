// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#ifndef RECOG_H
#define RECOG_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class hfrecog
{
	void * pModel;
	void * pData;
    std::vector<int> _input_dim;   // height, width, channels, num
    std::vector<int> _output_dim;  // height, width, channels, num

public:
	long create(const std::string& model_file, const std::string& passwd, int device);
	void destroy();
	long imrecog(cv::Mat cv_img, vector<float> & probs);
    const std::vector<int>& input_dim() const { return _input_dim;}
    const std::vector<int>& output_dim() const { return _output_dim;}
};


#endif
