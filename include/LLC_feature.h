#ifndef _LLC_H
#define _LLC_H

#include "encoding.h"
#include "kmeans.h"

class LLC_feature:public Encoding
{
public:
	LLC_feature(VlKMeans* podebook, hfrecog& recog, cv::Size size);
	virtual int extract_feature(const cv::Mat& image, std::vector<float>& fea, int step);
	virtual ~LLC_feature();

private:
	const static int KNN = 5;
    float * _codebook;
	int _nWords;
    VlKMeans* _pcodebook;

private:	
	void _kNearestNeighbor(float const* v1, std::vector<int> & index) const;
	//void _kNearestNeighborSSE(float const* v1, std::vector<int> & index) const;
};

#endif

