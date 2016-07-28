#include <emmintrin.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include "LLC_feature.h"

#undef max
#undef min
LLC_feature::LLC_feature(VlKMeans* pCodebook, hfrecog& recog, cv::Size size): Encoding(recog,size)
{
    _pcodebook = pCodebook;
	_codebook = (float*)pCodebook->centers;;
	_nWords = pCodebook->numCenters;
}

LLC_feature::~LLC_feature()
{
    if (_codebook != NULL) 
        delete [] _codebook;

    if (_pcodebook != NULL) 
        delete _pcodebook;
}


int LLC_feature::extract_feature(cv::Mat const & image, std::vector<float>& fea, int step = 8) 
{
    int nframes = _ftrNums;
    std::vector<cv::Point> point;      
    /* _extract_feature_cnn(image, step, nframes, point);
    if (nframes == 0) {
    return -1;
    }*/
 /*   std::fstream log_num("log_num.txt", std::ios::app);
    log_num<<nframes<<std::endl;
    log_num.close();
	*/
  /*  std::ofstream log("log_feature_out.txt", std::ios::out);
    for(int i = 0; i < nframes * _ftrDims; i++)
        log<<_feature[i]<<std::endl;*/
   
    CV_Assert(_pcodebook->dimension == _ftrDims);
    const float beta = 1e-4f;
    std::cout << "feature size : "<<nframes<<std::endl; 
	cv::Mat coeff(nframes, _nWords, CV_32FC1);
	for (int i = 0; i != nframes; ++i)
	{
		// input descriptor
		float const* query = _feature + _ftrDims*i;

		std::vector<int> KNNvect(KNN);
       // if (_ftrDims%4 != 0)
            _kNearestNeighbor(query, KNNvect);
		/*else
            _kNearestNeighborSSE(query, KNNvect);
        */     
		// encoding
		cv::Mat z(KNN, _ftrDims, CV_32FC1);
		for (int j = 0; j < KNN; ++j)
		{
			float const* B = _codebook + _ftrDims * KNNvect[j];

			for (int k = 0; k < _ftrDims; ++k)
			{
				z.at<float>(j, k) = B[k] - query[k];
			}
		}
		cv::Mat zt;
		cv::transpose(z, zt);
		cv::Mat C = z * zt;
		C = C + cv::Mat::eye(KNN, KNN, CV_32F) * beta * cv::trace(C)[0];
		cv::Mat w = C.inv() * cv::Mat::ones(KNN, 1, CV_32FC1);
		w = w / cv::sum(w)[0];
		for (int j = 0; j < KNN; ++j)
		{
			coeff.at<float>(i, KNNvect[j]) = w.at<float>(j, 0);
		}
	}

	// LLC_feature max pooling
	// PYRAMID IS FIXED TO SIZE [1x1, 2x2]
    bool spm  = 0;
    if(spm) {
        int imgH = image.rows;
        int imgW = image.cols;
	    std::vector<float> &pool = fea;//pool没有初始化 可能有bug!!!
	    pool.resize(5 * _nWords);
	    for (int i = 0; i < nframes; ++i)
	    {
		    double x = point[i].x;
		    double y = point[i].y;
		    int prow = (int)std::floor(y / (imgH / 2.0));
		    int pcol = (int)std::floor(x / (imgW / 2.0));
		    int offset = (1+2*prow+pcol)*_nWords;

		    for (int j = 0; j < _nWords; ++j)
		    {
    #undef max
			    pool[j] = std::max(pool[j], coeff.at<float>(i, j));
			    pool[j+offset] = std::max(pool[j+offset], coeff.at<float>(i, j));
		    }
	    }
    } else {
	    std::vector<float> &pool = fea;//pool没有初始化 可能有bug!!!
	    pool.resize(_nWords, 0);
	    for (int i = 0; i < nframes; ++i)
	    {	
		    for (int j = 0; j < _nWords; ++j)
		    {
			    pool[j] += coeff.at<float>(i, j);
		    }
	    }
        for (int j = 0; j < _nWords; ++j)
		{
			pool[j] /= nframes;
		}
    }
    nomalization(&fea[0], fea.size());
	return 0;
}

/*written by hele 2014/3/20*/
void LLC_feature::_kNearestNeighbor(float const* v1, std::vector<int> & index) const
{
	std::vector<float> tempD(KNN);
	for (size_t ii = 0; ii < KNN; ++ii)
	{
		index[ii] = 0;
		tempD[ii] = 10000000.0;
	}

	for (size_t jj = 0; jj < _nWords; ++jj)
	{

		float temp = 0;
		int loc = 0;
        float t=0;
		for (size_t kk = 0; kk < _ftrDims; ++kk)
		{	
            t = (v1[kk]-_codebook[_ftrDims*jj+kk]); 
            temp += t*t;
        }

		while (loc<KNN && temp>tempD[loc])
		{	
            ++loc;
        }
		if (loc<KNN)
		{
			for (size_t kk = KNN-1; kk > loc; --kk)
			{
				tempD[kk] = tempD[kk-1];
				index[kk] = index[kk-1];
			}
			tempD[loc] = temp;
			index[loc] = jj;
		}
	}
}

/*written by hele 2014/4/2 see*/
/*void LLC_feature::_kNearestNeighborSSE(float const* v1, std::vector<int> & index) const
{
	std::vector<float> tempD(KNN);
	for (size_t ii = 0; ii < KNN; ++ii)
	{
		index[ii] = 0;
		tempD[ii] = 10000000.0;
	}

	float const *pcodebook = _codebook;

    __m128 tsse;
	for (size_t jj = 0; jj < _nWords; ++jj)
	{

		float temp = 0;
		int loc = 0;

		__m128 tempsse = _mm_set_ps(0.0f,0.0f,0.0f,0.0f);
		float const *feature   = v1;

		for (size_t kk = 0; kk < _ftrDims/4; ++kk)
		{
            tsse = _mm_sub_ps(_mm_loadu_ps(feature+kk*4), _mm_loadu_ps(pcodebook+_ftrDims*jj+kk*4));
			tempsse = _mm_add_ps(tempsse, _mm_mul_ps(tsse, tsse));
		}							
		temp = temp + tempsse.m128_f32[0] + tempsse.m128_f32[1] + tempsse.m128_f32[2] + tempsse.m128_f32[3];

		while (loc<KNN && temp>tempD[loc]) {	
            ++loc;	
        }

		if (loc<KNN)
		{
			for (size_t kk = KNN-1; kk > loc; --kk)
			{
				tempD[kk] = tempD[kk-1];
				index[kk] = index[kk-1];
			}
			tempD[loc] = temp;
			index[loc] = jj;
		}

	}
}
*/
