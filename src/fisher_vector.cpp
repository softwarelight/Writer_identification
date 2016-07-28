#include "fisher_vector.h"
FisherVector::~FisherVector()
{
    if ( _pca != NULL) {
        delete _pca;
    }
}

int FisherVector::extract_feature(const cv::Mat& image, std::vector<float>& fea, int step) 
{
    int feature_num = _ftrNums;
    std::vector<cv::Point> point; 
    //_extract_feature_cnn(image, step, feature_num, point);
    
    float* cnn_feature;
    cv::Mat outdata;
    int fea_size;
    if (_use_pca) {
        cv::Mat indata(feature_num, _ftrDims, CV_32FC1, _feature);       
        _pca->project(indata, outdata);
        fea_size = outdata.cols;
        if (outdata.isContinuous())
            cnn_feature = (float*)outdata.data;
        else {
            std::cout<<"wrong! data is not continuous ! must change code"<<std::endl;
            exit(-1);
        }
    } else {
       cnn_feature = _feature;
       fea_size = _ftrDims;
    }

    for (int i = 0; i<feature_num; ++i){
        nomalization(cnn_feature + model._FeatureDim * i, model._FeatureDim);
    }

    CV_Assert(fea_size == model._FeatureDim);

    int fisher_feature_dim = 2 * model._FeatureDim * model._numComponents;

    bool _spm = 0;
    if ( !_spm) {       
        fea.resize(fisher_feature_dim);
        float* enc = &fea[0];
        vl_fisher_encode \
        (enc, VL_TYPE_FLOAT,
         model._means, model._FeatureDim, model._numComponents,     
         model._covariances, model._priors, cnn_feature, feature_num,
         VL_FISHER_FLAG_IMPROVED
         ) ;
        //nomalization(&fea[0], fea.size());//归一化不需要，vl_fisher里面已经做了归一化
    } else {
        // PYRAMID IS FIXED TO SIZE [1x1, 1x3]
        fea.resize(4 * fisher_feature_dim, 0);
	 
        vector<float> temp;
        temp.resize(fisher_feature_dim);
        float* enc = &temp[0];

        int imgH = image.rows;
        int imgW = image.cols;
        std::vector<int>  count(4, 0);
        count[0] = feature_num;
	    for (int i = 0; i < feature_num; ++i)
	    {            
            vl_fisher_encode \
            (enc, VL_TYPE_FLOAT,
            model._means, model._FeatureDim, model._numComponents,     
            model._covariances, model._priors, cnn_feature + i*fea_size, 1,
            VL_FISHER_FLAG_IMPROVED
            ) ;
		    double x = point[i].x;
		    double y = point[i].y;
         
		    int col2 = (int)std::floor(x / (imgW / 3.0));
		    int offset = (1+col2)*fisher_feature_dim;
            count[1+col2]++;

		    for (int j = 0; j < fisher_feature_dim; ++j)
		    {
			    fea[j] += temp[j];			  
                fea[j+offset] += temp[j];                
		    }
	    }

        for (int i = 0; i<4; i++) {
            int offset = i*fisher_feature_dim;
            for (int j = 0; j < fisher_feature_dim; ++j)
		    {             
			    fea[j + offset] /= count[i];
		    }
        }
        //nomalization(&fea[0], fea.size()); //归一化不需要，vl_fisher里面已经做了归一化
    }
    //else {
    //    // PYRAMID IS FIXED TO SIZE [1x1, 2x2]
    //    fea.resize(5 * fisher_feature_dim, 0);
	 
    //    vector<float> temp;
    //    temp.resize(fisher_feature_dim);
    //    float* enc = &temp[0];

    //    int imgH = image.rows;
    //    int imgW = image.cols;
    //    std::vector<int>  count(5, 0);
    //    count[0] = feature_num;
	   // for (int i = 0; i < feature_num; ++i)
	   // {            
    //        vl_fisher_encode \
    //        (enc, VL_TYPE_FLOAT,
    //        model._means, model._FeatureDim, model._numComponents,     
    //        model._covariances, model._priors, cnn_feature + i*fea_size, 1,
    //        VL_FISHER_FLAG_IMPROVED
    //        ) ;
		  //  double x = point[i].x;
		  //  double y = point[i].y;
    //      
		  //  int row2 = (int)std::floor(y / (imgH / 2.0));
		  //  int col2 = (int)std::floor(x / (imgW / 2.0));
		  //  int offset = (1+2*row2+col2)*fisher_feature_dim;
    //        count[1+2*row2+col2]++;

		  //  for (int j = 0; j < fisher_feature_dim; ++j)
		  //  {
			 //   fea[j] += temp[j];			  
    //            fea[j+offset] += temp[j];                
		  //  }
	   // }

    //    for (int i = 0; i<5; i++) {
    //        int offset = i*fisher_feature_dim;
    //        for (int j = 0; j < fisher_feature_dim; ++j)
		  //  {             
			 //   fea[j + offset] /= count[i];
		  //  }
    //    }
    //}
    //else {
    //    // PYRAMID IS FIXED TO SIZE [1x1, 1x3, 2x2]
    //    fea.resize(8 * fisher_feature_dim, 0);
	 
    //    vector<float> temp;
    //    temp.resize(fisher_feature_dim);
    //    float* enc = &temp[0];

    //    int imgH = image.rows;
    //    int imgW = image.cols;
    //    std::vector<int>  count(8, 0);
    //    count[0] = feature_num;
	   // for (int i = 0; i < feature_num; ++i)
	   // {            
    //        vl_fisher_encode \
    //        (enc, VL_TYPE_FLOAT,
    //        model._means, model._FeatureDim, model._numComponents,     
    //        model._covariances, model._priors, cnn_feature + i*fea_size, 1,
    //        VL_FISHER_FLAG_IMPROVED
    //        ) ;
		  //  double x = point[i].x;
		  //  double y = point[i].y;
    //        int col1 = (int)std::floor(x / (imgW / 3.0));
    //        int offset1 = (1+col1)*fisher_feature_dim;
    //        count[1+col1]++;

		  //  int row2 = (int)std::floor(y / (imgH / 2.0));
		  //  int col2 = (int)std::floor(x / (imgW / 2.0));
		  //  int offset2 = (4+2*row2+col2)*fisher_feature_dim;
    //        count[4+2*row2+col2]++;

		  //  for (int j = 0; j < fisher_feature_dim; ++j)
		  //  {
			 //   fea[j] += temp[j];
			 //   fea[j+offset1] += temp[j];
    //            fea[j+offset2] += temp[j];                
		  //  }
	   // }

    //    for (int i = 0; i<8; i++) {
    //        int offset = i*fisher_feature_dim;
    //        for (int j = 0; j < fisher_feature_dim; ++j)
		  //  {             
			 //   fea[j + offset] /= count[i];
		  //  }
    //    }
    //}
    return 0;
}
