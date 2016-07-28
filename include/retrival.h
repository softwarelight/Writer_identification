#ifndef RETRIVAL_H
#define RETRIVAL_H

#include <string>
#include <fstream>

#include "recog.hpp"
#include "encoding.h"   

#define PCA_MEAN    "mean"
#define PCA_EIGEN_VECTOR    "eigen_vector"
#define PCA_EIGEN_VALUE     "eigen_value"

#define SUCCESS 0
#define L2_DIST 0
#define COSINE_DIST 1
#define METRIC_DIST 2
#define CHI_DIST 3
#define WEI_L2_DIST 4
#define MANHATTAN_DIST 5
//#define HFRZ_DEBUG 

extern int64_t extract_feature_time;
extern int64_t features_pca_time;
extern int64_t query_image_time_once;

struct Imdb {
    int size;
    int dimension;
    std::vector<float *> imdb;
    std::vector<std::string> image_path;
    std::vector<int> image_class;
};

//class LLC_feature;
class HfrzRetrival{
  
public:

    HfrzRetrival( cv::Size size, bool pca, int dist_method, int code_method, bool device = 1);

    ~HfrzRetrival();

    int load_model(std::string model_path, int step);

    //从gellery生成图片特征库，use_pca = 0 表示不用pca
    int get_imdb_from_gallery(std::string src, std::string dst);

    int	get_imdb_with_sub_dir(std::string patch_dir, std::string pic_dir, std::string dst);
    
    int get_standard_deviation_with_sub_dir(std::string patch_dir, std::string dst);

    int load_standard_deviation(std::string src);

    int load_imdb(std::string src);
   
	int query_image(const cv::Mat &img, const bool ust_pca, const int topN,\
        std::vector<std::pair<int,float> > &topResult);

    int query_image_patch(std::string &patch_dir, std::vector<std::pair<int,float> > &topResult);

    cv::Size get_recog_size() {
        return _recog_image_size;
    }

    const Imdb& get_imdb() {
        return _image_db;
    }

    float get_distance(const float *f1, const float *f2, const int dimension);
    inline float get_distance_L2(const float *f1, const float *f2, const int dimension);
    inline float get_distance_cosine(const float *f1, const float *f2, const int dimension);
    float get_distance_metric(const float *f1, const float *f2, const int dimension);
    inline float get_distance_chi(const float *f1, const float *f2, const int dimension);
    inline float get_distance_manhattam(const float *f1, const float *f2, const int dimension);
private:
    
   
    int load_pca_matrix(std::string src);

    HfrzRetrival(HfrzRetrival& retrival);

    HfrzRetrival operator =(HfrzRetrival& retrival); 

//private:
public:

    hfrecog _recog;
    std::string _deploy_path, _param_path, _mean_path;
    cv::Size _recog_image_size;
    std::vector<float> _standard_deviation;
    Encoding *_fea_handler;
   
    Imdb _image_db;
    
    cv::Mat _pca_matrix;    
    cv::PCA *_pca;

    bool _use_pca, _device;  // 1 GPU 0 CPU 
    int _code_method; //0-> don't coding 1->llc  2->fisher vector
    int _dist_method; // 0->L2, 1->cosine distance, 2->metric
    int _llc_step;

    int _feature_size;
    float *_f1, *_f2;  // 计算向量距离时的缓存空间

};

inline float HfrzRetrival::get_distance_cosine(const float *f1, const float *f2, const int dimension)
{
    float distance = 0;
    float d1 = 0;
    float d2 = 0;
    float inner_product = 0;

    for (size_t i = 0; i < dimension; i++) {
       inner_product += f1[i]*f2[i];
       d1 += f1[i]*f1[i];
       d2 += f2[i]*f2[i];
    }
    distance = inner_product / sqrt(d1) / sqrt(d2);
    return distance;
}

inline float HfrzRetrival::get_distance_chi(const float *f1, const float *f2, const int dimension)
{
    float L2_distance = 0;
    float temp = 0;
    float temp2;
    for (size_t i = 0; i < dimension; i++) {
        temp = f1[i] - f2[i];
        temp *= temp;
        temp2 = f1[i] + f2[i];
        if (temp2 != 0)
            L2_distance += temp/(f1[i] + f2[i]);
    }
    return L2_distance;
}


inline float HfrzRetrival::get_distance_manhattam(const float *f1, const float *f2, const int dimension)
{
    float L2_distance = 0;
    float temp = 0;
    for (size_t i = 0; i < dimension; i++) {
        temp = abs(f1[i] - f2[i]);        
        L2_distance += temp;
    }
    return L2_distance;
}
#endif
