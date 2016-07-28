#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <stdint.h>

#include "FsHelpers.h"
#include "retrival.h"
#include "result_sort.h"
#include "LLC_feature.h"
#include "code_book.h"
#include "fisher_vector.h"
#include "def.h"

int64_t extract_feature_time = 0;
int64_t features_pca_time = 0;
int64_t query_image_time_once = 0;

HfrzRetrival::HfrzRetrival( cv::Size size, bool pca, int dist_method, int code_method, bool device)\
    :_device(device), _recog_image_size(size), _use_pca(pca), _dist_method(dist_method), _code_method(code_method)
{
    _llc_step = 8;

	_image_db.size = 0;
	_image_db.dimension = 0;
    
    _feature_size = 0;
    _f1 = NULL; 
    _f2 = NULL; 

    _fea_handler = NULL;
}

HfrzRetrival::~HfrzRetrival()
{
    //if (_recog != NULL) {
        _recog.destroy();       
    //}
    for (size_t i = 0; i < _image_db.imdb.size(); i++) {
        delete []_image_db.imdb[i];
    }

    if ( _f1 != NULL) {
        delete []_f1;
    }
    if ( _f2 != NULL) {
        delete []_f2;
    }
}

int HfrzRetrival::load_model(std::string model_dir,int step)
{
    _llc_step = step;

    // 特征提取初始化

    _recog.create(FsHelpers::CombinePath(model_dir,"md.dat"), PASS_WORD, _device);
   
    // 导入矩阵参数
    if(_dist_method == 2) {
        std::string metric_path = model_dir + '/' + "matrix.txt";
        load_pca_matrix(metric_path);
    }

    // pca降维初始化
    if (_use_pca) {
        std::string pca_path = model_dir + '/' + "pca_config.xml";
        _pca = new cv::PCA();
        cv::FileStorage cvfs(pca_path, cv::FileStorage::READ);

        if (!cvfs.isOpened()) {
            std::cout<<"pca file can't open correctly!"<<std::endl;
            return -1;
        }
        cvfs[PCA_MEAN] >> _pca->mean;
        cvfs[PCA_EIGEN_VECTOR] >> _pca->eigenvectors;
        cvfs.release();
    }

    if(_code_method == 1) {
        std::string codebook_path = model_dir + '/' + "codebook_llc.dat";
        Codebook *pc= new Codebook();
        VlKMeans* pv = NULL;
        pc->loadCodebook(codebook_path, &pv);
        _fea_handler = new LLC_feature(pv, _recog, _recog_image_size);
        delete pc;
    } else if(_code_method == 2){
        std::string codebook_path = model_dir + '/' + "gmm_model.dat";
        FisherVector *temp = new FisherVector(_recog, _recog_image_size);

        if(temp->model.loadCodebook(codebook_path) != 0 ) {
            delete temp;
            return -1;
        }
        
        std::string fisher_pca_path = FsHelpers::CombinePath(model_dir, "fisher_pca.xml");
        if(FsHelpers::Exists(fisher_pca_path)) {
            temp->_use_pca = 1;
            temp->_pca = new cv::PCA();
            cv::FileStorage cvfs(fisher_pca_path, cv::FileStorage::READ);

            if (!cvfs.isOpened()) {
                std::cout<<"pca file can't open correctly!"<<std::endl;
                return -1;
            }
            cvfs[PCA_MEAN] >> temp->_pca->mean;
            cvfs[PCA_EIGEN_VECTOR] >> temp->_pca->eigenvectors;
            cvfs.release();
        }
        _fea_handler = temp;
    }
    return 0;
}

int HfrzRetrival::get_imdb_from_gallery(std::string src, std::string dst)
{
    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    } 

    vector<string> files;
    FsHelpers::GetFilesHasExtension(src, files, ".jpg",false);
    FsHelpers::GetFilesHasExtension(src, files, ".bmp",false);

    int sz = files.size();
    fout.write((char*)&sz,sizeof(sz));

    for (size_t j=0; j<files.size(); j++) {
        cv::Mat img = cv::imread(files[j]);     

        std::string img_name = FsHelpers::GetFileName(files[j]);
        std::vector<float> features;
        if(!_code_method) {
            cv::Mat im_resized(get_recog_size(), img.type());
            cv::resize(img, im_resized, im_resized.size(),0,0, CV_INTER_CUBIC);
            std::cout<<j<<":"<<_recog.imrecog(im_resized,features)<<std::endl; 
        } else {
            std::cout<<j<<":"<<_fea_handler->extract_feature(img, features, _llc_step)<<";"<<img_name<<std::endl;
        }
        cv::Mat f_mat(1, features.size(), CV_32FC1, &features[0]);        

        cv::Mat f_pca;
        if (_use_pca){
           /* outdata.create(rows, index, CV_32FC1);    
            pca->project(indata, outdata);*/
            f_pca = _pca->project(f_mat);
        } else {
            f_pca = f_mat;
        }

        int dim = f_pca.cols;
        if (j == 0) fout.write((char*)&dim, sizeof(int)); 

        int path_size = files[j].size();
        const char *path = files[j].c_str();

        // test writer icdar 2013 dataset
#ifdef TEST_ICDAR
        int id1 = atoi(img_name.substr(0,3).c_str());
        int id2 = (img_name[4] - '0')*1000;
        int image_class = id1 + id2;
#endif 

        //test other       
#ifndef TEST_ICDAR
        int image_class = atoi((img_name.substr(0,5)).c_str());
#endif        

        fout.write((char*)&path_size, sizeof(int));
        fout.write(path, sizeof(char)*(path_size+1));
        fout.write((char*)&image_class, sizeof(int));

        float* data= f_pca.ptr<float>(0);  
        fout.write((char*)data, sizeof(float) * dim);
    }        

    fout.close();  
    return 0;
}


int HfrzRetrival::get_imdb_with_sub_dir(std::string patch_dir, std::string pic_dir, std::string dst)
{
    std::vector<std::string> directories;
    FsHelpers::GetDirectories(patch_dir,directories);

    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"matrix file can't open correctly!"<<std::endl;
        return -1;
    }

#ifdef TWO_LAYER_FEATURE
    std::string model_path = "models_writer";
    hfrecog Recog;
    Recog.create(model_path, PASS_WORD, 1);
#endif

    int sz = directories.size();
    fout.write((char*)&sz,sizeof(sz));

    for (size_t i = 0; i < sz;  i++) {         
        std::cout<<i<<'\t'<<directories[i]<<" of "<<sz<<std::endl;
        vector<string> files;
        std::string img_name = FsHelpers::GetFileName(directories[i]);
        FsHelpers::GetFilesHasExtension(directories[i], files, ".jpg",false);
        FsHelpers::GetFilesHasExtension(directories[i], files, ".png",false);
        FsHelpers::GetFilesHasExtension(directories[i], files, ".bmp",false);

        int nums = files.size();
        _fea_handler->_ftrNums = files.size();

        for (size_t j=0; j<files.size(); j++) {
            cv::Mat img = cv::imread(files[j]);
            std::vector<float> feature2, features;         
            _recog.imrecog(img,features);

#ifdef TWO_LAYER_FEATURE
            Recog.imrecog(img, feature2);
            features.insert(features.end(), feature2.begin(), feature2.end());
#endif
            if (i == 0 && j == 0) {
                if(_fea_handler->_feature == NULL) {
                    _fea_handler->_ftrDims = features.size();
           //         cout<<"feature size "<<features.size()<<endl;
           //         getchar();

                    if(_fea_handler->_capacity < nums)
                        _fea_handler->_capacity *= 2;
                    _fea_handler->_feature = (float*)malloc(sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims));
                } else if(_fea_handler->_capacity < nums) {
                    float* temp = (float*)malloc(sizeof(float)*(_fea_handler->_capacity) * (_fea_handler->_ftrDims)*2);
                    memcpy(temp, (_fea_handler->_feature), sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims)); 
                    delete []_fea_handler->_feature;
                    _fea_handler->_feature = temp;
                    _fea_handler->_capacity *=2;
                }         
            }
            memcpy(_fea_handler->_feature + j*_fea_handler->_ftrDims, &features[0], (_fea_handler->_ftrDims) * sizeof(float));
        }  
        std::vector<float> fisher_features;
        cv::Mat temp;
        _fea_handler->extract_feature(temp, fisher_features, _llc_step);

        if(i == 0) {
            int feature_size = fisher_features.size();
            fout.write((char*)&feature_size, sizeof(int));  
        }

        //string img_path_t = pic_dir + '/' + img_name + ".jpg";
        string img_path_t = img_name + ".jpg";

        int path_size = img_path_t.size();
        const char *path = img_path_t.c_str();
        fout.write((char*)&path_size, sizeof(int));
        fout.write(path, sizeof(char)*(path_size+1));

        // test writer icdar 2013 dataset
#ifdef TEST_ICDAR  
        int id1 = atoi(img_name.substr(0,3).c_str());
        int id2 = (img_name[4] - '0')*1000;
        int image_class = id1 + id2;
#else
    #ifdef TEST_CVL
        int id1 = atoi(img_name.substr(0,4).c_str());
        int id2 = (img_name[5] - '0')*10000;
        int image_class = id1 + id2;
    #else   //test other
        int image_class = atoi((img_name.substr(0,5)).c_str());
    #endif
#endif 
             
        fout.write((char*)&image_class, sizeof(int));     
        fout.write((char*)&fisher_features[0], sizeof(float) * fisher_features.size());
    }
    
    fout.close();  
    return 0;
}

int HfrzRetrival::load_standard_deviation(std::string src)
{
    std::ifstream ifp(src.c_str(), std::ios::binary);
    int fea_dim;
    ifp.read((char *)&fea_dim, sizeof(float));
    std::cout<< "fesher feature size: " << fea_dim << std::endl;
    _standard_deviation.resize(fea_dim);
    ifp.read((char *)&_standard_deviation[0], sizeof(float) * fea_dim);
    ifp.close();
    return 0;
}

int HfrzRetrival::get_standard_deviation_with_sub_dir(std::string patch_dir, std::string dst)
{
    std::vector<std::string> directories;
    FsHelpers::GetDirectories(patch_dir,directories);

    int sz = directories.size(); 
    int fea_size;
    std::vector<std::vector<float>>  features;
    for (size_t i = 0; i < sz;  i++) {         
        std::cout<<i<<std::endl;
        vector<string> files;
        std::string img_name = FsHelpers::GetFileName(directories[i]);
        FsHelpers::GetFilesHasExtension(directories[i], files, ".jpg",false);
        FsHelpers::GetFilesHasExtension(directories[i], files, ".png",false);

        int nums = files.size();
        _fea_handler->_ftrNums = files.size();

        for (size_t j=0; j<files.size(); j++) {
            cv::Mat img = cv::imread(files[j]);
            std::vector<float> feature2, features;         
            _recog.imrecog(img,features);

            if (i == 0 && j == 0) {
                if(_fea_handler->_feature == NULL) {
                    _fea_handler->_ftrDims = features.size();
                    if(_fea_handler->_capacity < nums)
                        _fea_handler->_capacity *= 2;
                    _fea_handler->_feature = (float*)malloc(sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims));
                } else if(_fea_handler->_capacity < nums) {
                    float* temp = (float*)malloc(sizeof(float)*(_fea_handler->_capacity) * (_fea_handler->_ftrDims)*2);
                    memcpy(temp, (_fea_handler->_feature), sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims)); 
                    delete []_fea_handler->_feature;
                    _fea_handler->_feature = temp;
                    _fea_handler->_capacity *=2;
                }         
            }
            memcpy(_fea_handler->_feature + j*_fea_handler->_ftrDims, &features[0], (_fea_handler->_ftrDims) * sizeof(float));
        }  

        std::vector<float> fisher_feature;        
        cv::Mat temp;
        std::cout<<i<<":"<<_fea_handler->extract_feature(temp, fisher_feature, _llc_step)<<";"\
            << img_name <<std::endl; 

        if(i==0)
            fea_size = fisher_feature.size();

        features.push_back(fisher_feature);
    }   
    int fea_num =  features.size();
    std::vector<float> mean(fea_size, 0);
    std::vector<float> sd(fea_size, 0);
    for (int i = 0; i < fea_num; i++) {    
        auto &v = features[i];
        for(int j = 0; j<fea_size; j++) {
            mean[j] += v[j];
        }
    }

    for(int j = 0; j<fea_size; j++) {
        mean[j] /= fea_num;
    }

    float temp;
    for (int i = 0; i < fea_num; i++) {    
        auto &v = features[i];
        for(int j = 0; j<fea_size; j++) {
            temp = v[j] - mean[j];
            sd[j] += (temp * temp);
        }
    }
    for(int j = 0; j<fea_size; j++) {
       sd[j] = sqrt(sd[j] / fea_num);
    }

    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"matrix file can't open correctly!"<<std::endl;
        return -1;
    }

    fout.write((char*)&fea_size, sizeof(int));
    fout.write((char*)&sd[0], sizeof(float)*fea_size);
    fout.close();  
    return 0;
}

int HfrzRetrival::load_imdb(string src)
{
    std::ifstream ifp(src.c_str(), std::ios::binary);
    ifp.read((char *)&_image_db.size, sizeof(float));
    ifp.read((char *)&_image_db.dimension, sizeof(float));

    for (size_t i = 0; i < _image_db.size; i++) {
        float * temp = new float[_image_db.dimension];
        int path_size, image_class, image_id;
        
        ifp.read((char*)&path_size, sizeof(int));
        char *path = new char[path_size+1];
        ifp.read((char*)path, sizeof(char)*(path_size+1));
        ifp.read((char*)&image_class, sizeof(int));
        ifp.read((char*)temp, sizeof(float)*_image_db.dimension);

        _image_db.imdb.push_back(temp);
        _image_db.image_class.push_back(image_class);
        _image_db.image_path.push_back(string(path));
        delete []path;
    }
    return SUCCESS;
}

int HfrzRetrival::load_pca_matrix(string src)
{
    std::fstream f_pca(src.c_str());
    if (!f_pca.is_open()) {
        std::cout<<"file:"<<"\""<<src<<"\""<<"cannot open correctly!"<<std::endl;
        return -1;
    }
    size_t dim; 
    std::string line;
    getline(f_pca,line);
    std::istringstream text(line);
    text >> dim;

    _pca_matrix.create(dim, dim, CV_64FC1);
   
    for (size_t i = 0; i < dim; i++) {
        getline(f_pca,line);
        std::istringstream text(line);

        for (size_t j = 0 ; j < dim; j++) {
            double temp ;
            text >> temp;
            _pca_matrix.at<double>(i,j) = temp;
        }
    }

#ifdef HFRZ_DEBUG 
    std::ofstream fp("log.txt");
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0 ; j < dim; j++) {            
            fp << std::setiosflags(std::ios::fixed) << std::setprecision(10);
            fp << _pca_matrix.at<double>(i,j) << " ";
        }
        fp << std::endl;
    }
    fp.close();
#endif

    return 0;
}

int HfrzRetrival::query_image(const cv::Mat &img, const bool use_pca,\
    const int rank, std::vector<std::pair<int,float> > &topResult)
{
    if (_image_db.size == 0)    return -1;

    std::vector<float> scores;     
    std::vector<float> features;

    uint64_t feature_time = 0;{
        string proc_desc;
        if(!_code_method) {
            cv::Mat im_resized(get_recog_size(), img.type());
            cv::resize(img, im_resized, im_resized.size(),0,0, CV_INTER_CUBIC);
            _recog.imrecog(im_resized,features); 
        } else {
            _fea_handler->extract_feature(img, features, _llc_step);
        }
    }
    extract_feature_time += feature_time;

    float *f_feature = &features[0];
    cv::Mat f_mat(1, features.size(), CV_32FC1, f_feature);

    cv::Mat f_pca;
    if (use_pca) {
        f_pca = _pca->project(f_mat);
    } else {
        f_pca = f_mat;
    }

    int dim = f_pca.cols;
    float *f1 = f_pca.ptr<float>(0);   

    for (size_t j = 0; j < _image_db.size; j++)	{ 
        assert(_image_db.dimension == dim);
        float *f2 =  _image_db.imdb[j];        

        float dist = get_distance(f1, f2, dim);
        if(dist == -1)
            return -1;
        scores.push_back(dist);
    }

    float max = 0;
    float min = 100000;
    for (size_t i = 0; i < scores.size(); i++) {
        scores[i] = scores[i] + 0.00001;
        scores[i] = 1/scores[i];

        if(scores[i] > max) {
            max = scores[i];
        }

        if(scores[i] < min) {
            min = scores[i];
        }
    }

    float inter = max - min;
    for (int i=0; i <scores.size(); i++) {
        scores[i] = (scores[i] - min) / inter;
    }

    result_sort(scores, rank, topResult);
    return SUCCESS;
}


int HfrzRetrival::query_image_patch(std::string &patch_dir, std::vector<std::pair<int,float> > &topResult)
{
    if (_image_db.size == 0)    return -1;
    std::vector<float> scores;     

    vector<string> files;
    FsHelpers::GetFilesHasExtension(patch_dir, files, ".jpg", true);
    FsHelpers::GetFilesHasExtension(patch_dir, files, ".bmp",false);
    FsHelpers::GetFilesHasExtension(patch_dir, files, ".png",false);

    int nums = files.size();
    _fea_handler->_ftrNums = nums;

    for (size_t j=0; j<files.size(); j++) {
        cv::Mat img = cv::imread(files[j]);
        std::vector<float> features;         
        _recog.imrecog(img,features);          

        if (j == 0 && _fea_handler->_feature == NULL) {         
            _fea_handler->_ftrDims = features.size();
            if(_fea_handler->_capacity < nums) {
                _fea_handler->_capacity *= 2;
            }
            _fea_handler->_feature = (float*)malloc(sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims));
        } else if(_fea_handler->_capacity < nums) {
            float* temp = (float*)malloc(sizeof(float)*(_fea_handler->_capacity) * (_fea_handler->_ftrDims)*2);
            memcpy(temp, (_fea_handler->_feature), sizeof(float)*(_fea_handler->_capacity)*(_fea_handler->_ftrDims)); 
            delete []_fea_handler->_feature;
            _fea_handler->_feature = temp;
            _fea_handler->_capacity *=2;
        }            
        memcpy(_fea_handler->_feature + j*_fea_handler->_ftrDims, &features[0], (_fea_handler->_ftrDims) * sizeof(float));
    }  
    std::vector<float> fisher_features;
    cv::Mat temp;
    std::string img_name = FsHelpers::GetFileName(patch_dir);
    _fea_handler->extract_feature(temp, fisher_features, _llc_step);    
   
    float *f1 = &fisher_features[0];
    int dim = fisher_features.size();

    for (size_t j = 0; j < _image_db.size; j++)	{ 
        assert(_image_db.dimension == dim);
        float *f2 =  _image_db.imdb[j];        

        float dist;
        dist = get_distance(f1, f2, dim);
        if (dist == -1)
            return -1;
        scores.push_back(dist);
    }

    float max = 0;
    float min = 100000;
    for (size_t i = 0; i < scores.size(); i++) {
        if (_dist_method != COSINE_DIST) {
            scores[i] = scores[i] + 0.00001;
            scores[i] = 1/scores[i];
        }

        if(scores[i] > max) {
            max = scores[i];
        }

        if(scores[i] < min) {
            min = scores[i];
        }
    }

    float inter = max - min;
    for (int i=0; i <scores.size(); i++) {
        scores[i] = (scores[i] - min) / inter;
    }

    result_sort(scores, _image_db.size, topResult);
    return SUCCESS;
}


float HfrzRetrival::get_distance(const float *f1, const float *f2, const int dim)
{
    float dist;
    if (_dist_method == METRIC_DIST)
    {
        dist = get_distance_metric(f1, f2, dim);
    } else if ( _dist_method == L2_DIST || _dist_method == WEI_L2_DIST ) {
        dist = get_distance_L2(f1, f2, dim);
    } else if (_dist_method == COSINE_DIST) {
        dist = get_distance_cosine(f1, f2, dim);
    } else if (_dist_method == CHI_DIST) {
        dist = get_distance_chi(f1, f2, dim);
    } else if (_dist_method == MANHATTAN_DIST) {
        dist = get_distance_manhattam(f1, f2, dim);
    } else {
        std::cout<<"error distance type!"<<std::endl;
        return -1;
    }
    return dist;
}

inline float HfrzRetrival::get_distance_L2(const float *f1, const float *f2, const int dimension)
{
    float L2_distance = 0;
    float temp = 0;
    for (size_t i = 0; i < dimension; i++) {       
        temp = f1[i] - f2[i]; 
        if(WEI_L2_DIST == _dist_method)
            temp /= _standard_deviation[i];
        temp *= temp;
        L2_distance += temp;
    }
    return L2_distance;
}


float HfrzRetrival::get_distance_metric(const float *f1, const float *f2, const int dimension)
{
    assert(dimension == _pca_matrix.cols);

    if (_feature_size < dimension) {
        _feature_size = dimension;

        if(_f1 !=NULL) delete []_f1;
        if(_f2 !=NULL) delete []_f2;

        _f1 = new float[dimension];
        _f2 = new float[dimension];
    }

    memset(_f1, 0, sizeof(float) * dimension);
    memset(_f2, 0, sizeof(float) * dimension);

    for (size_t i = 0; i < dimension; i++) {
        _f1[i] = f1[i] - f2[i];
    }   

    for (size_t i = 0; i < dimension; i++) {
        for (size_t j = 0; j < dimension; j++) {
            float temp = _pca_matrix.at<double>(j,i);
            _f2[i] += _f1[j] * temp;
        }
    }

    double sum = 0;

    for (size_t i = 0; i < dimension; i++) {
        sum += _f1[i] * _f2[i];
    }

    return sum;
}

