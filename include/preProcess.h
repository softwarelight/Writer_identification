/*************************************************************************
	> File Name: preProcess.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Thu 07 Jul 2016 12:00:56 PM CST
 ************************************************************************/

#include<iostream>
#include<opencv2/core/core.hpp>

void icvprCcaByTwoPass(const cv::Mat& binImage, cv::Mat& labelImg) ; 

void icvprLabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg) ;  
