#include <fstream>
#include <vector>
#include "fisher_model.h"
#include "FsHelpers.h"
#include "utilities.h"
#include "def.h"
FisherModel::FisherModel()
{
    _numComponents = 256;
    _FeatureDim = 100;
    _patch_num = 0;
    _descriptors = NULL;
    _gmm = NULL;

    _means = NULL;
    _covariances = NULL;
    _priors = NULL;
}

FisherModel::FisherModel(int num_comp)
{
    _numComponents = num_comp;
    _FeatureDim = 100;
    _patch_num = 0;
    _descriptors = NULL;
    _gmm = NULL;

    _means = NULL;
    _covariances = NULL;
    _priors = NULL;
}

int FisherModel::get_features(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst)
{
    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    }

    std::vector<std::string> files;
    FsHelpers::GetFilesHasExtension(src, files, ".jpg",true);
    _patch_num = files.size();
    std::cout<<"total patch: "<<_patch_num<<std::endl;
    //_patch_num = 100;
    fout.write((char*)&_patch_num, sizeof(_patch_num));
    int width = patch_size.width;
    int height = patch_size.height;    

    vector<float> feature;
    int feature_size;

    std::string model_path = "models_writer";

#ifdef TWO_LAYER_FEATURE
    hfrecog Recog;
    Recog.create(model_path, PASS_WORD, 1);
#endif

    for (size_t j=0; j<_patch_num; j++) {
        std::cout<<j<<std::endl;
        cv::Mat img = cv::imread(files[j]);
        int channel = img.channels();

        feature.clear();
        retrival._recog.imrecog(img,feature);

#ifdef TWO_LAYER_FEATURE
        std::vector<float> feature2;         
        Recog.imrecog(img, feature2);
        feature.insert(feature.end(), feature2.begin(), feature2.end());
#endif

        if (j == 0) {
            feature_size = feature.size();
            fout.write((char*)&feature_size, sizeof(int)); 
            std::cout<<"_FeatureDim: "<<feature_size<<std::endl;
            _descriptors = (float*)malloc(_patch_num*feature_size*sizeof(float)); 
          
           //_descriptors = new(std::nothrow) float[_patch_num*feature_size*sizeof(float)];
           if (_descriptors == NULL) {
               std::cout<< "malloc memory error!" <<std::endl;
               return -1;
           }
            //_descriptors = new float[100*feature_size*sizeof(float)];
            _FeatureDim = feature_size;
           
        }

        memcpy(_descriptors+j*feature_size, &feature[0], sizeof(float)*feature_size);

        std::string name = FsHelpers::GetFileName(files[j]);
        int id1 = atoi(name.substr(0,3).c_str());
        int id2 = (name[4] - '0')*1000;
        int id = id1 + id2;
        fout.write((char*)&id, sizeof(int));     
        fout.write((char*)&feature[0], sizeof(float) * feature.size());
    }        
    fout.close();  
    return 0;
}

int FisherModel::load_features(std::string src)
{
    std::ifstream ifp(src.c_str(), std::ios::binary);
    ifp.read((char *)&_patch_num, sizeof(float));
    ifp.read((char *)&_FeatureDim, sizeof(float));
    _descriptors = (float*)malloc(_patch_num*_FeatureDim*sizeof(float));
    //_patch_num = 5000;
    for (size_t i = 0; i < _patch_num; i++) {
        int image_class;       
        ifp.read((char*)&image_class, sizeof(int));
        ifp.read((char*)(_descriptors + i*_FeatureDim), sizeof(float)*_FeatureDim);
    }
    return SUCCESS;
}

void FisherModel::nomalization(float *data, int size) const
{
    double norm = 0.0f;
    for (size_t i = 0; i < size; ++i)
        norm += data[i] * data[i];
    norm = sqrtf(norm);

    for (size_t i = 0; i < size; ++i)
        data[i] /= norm;
}

int FisherModel::computeCodebook(float pca_ratio, std::string dst)
{
    float* pdata = _descriptors;
    cv::Mat outdata;
    if(pca_ratio > 0) {
        cv::Mat indata(_patch_num, _FeatureDim, CV_32FC1, _descriptors);        
        mat_pca(indata, dst, pca_ratio, outdata);
        _FeatureDim = outdata.cols;    

        if(!outdata.isContinuous()) {        
            std::cout<<"wrong! data is not continuous ! must change code"<<std::endl;
        }
        pdata = (float *)outdata.data;
    }
    
    for (int i = 0; i<_patch_num; ++i){
        nomalization(pdata + _FeatureDim * i, _FeatureDim);
    }

    _gmm = vl_gmm_new (VL_TYPE_FLOAT, _FeatureDim, _numComponents) ;
    if (_gmm == NULL)
        return -1;

    // create a new instance of a GMM object for float data
    vl_gmm_set_max_num_iterations (_gmm, 100);
    // set the initialization to random selection
    vl_gmm_set_initialization (_gmm,VlGMMRand);
    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster (_gmm, pdata, _patch_num);

    return 0;
}

bool FisherModel::saveCodebook(const std::string dst)
{
    std::ofstream fp(dst);
    if (!fp.is_open())
    {
        printf("Failed to open codebook file %s\n", dst.c_str());
        return false;
    }

    // get the means, covariances, and priors of the GMM 
    float* means = (float *)vl_gmm_get_means(_gmm);
    float* covariances = (float *)vl_gmm_get_covariances(_gmm);
    float* priors = (float *)vl_gmm_get_priors(_gmm);

    fp << _numComponents << " " << _FeatureDim << std::endl;
    int sz = _numComponents * _FeatureDim;

    for(int i = 0; i < sz; i++)
        fp << means[i] << " ";
    fp << std::endl;
    for(int i = 0; i < sz; i++)
        fp << covariances[i] << " ";
    fp << std::endl;
    for(int i = 0; i < _numComponents; i++)
        fp << priors[i] << " ";
    fp << std::endl;
    fp.close();
    return true;
}

bool FisherModel::loadCodebook(const std::string codebookFile)
{
    std::ifstream fp(codebookFile);
    if (!fp.is_open())
    {
        printf("Failed to load codebook file %s\n", codebookFile.c_str());
        return -1;
    }
    fp >> _numComponents >> _FeatureDim;
    int sz = _numComponents * _FeatureDim;

    _means = (float *)malloc(sizeof(float)*sz);
    _covariances = (float *)malloc(sizeof(float)*sz);
    _priors = (float*)malloc(sizeof(float)*_numComponents);

    for(int i = 0; i < sz; i++)
        fp >> _means[i];
    for(int i = 0; i < sz; i++)
        fp >> _covariances[i];  
    for(int i = 0; i < _numComponents; i++)
        fp >> _priors[i];

    fp.close();
    return 0;
}


VlGMM* FisherModel::get_codebook()
{
    return _gmm;
}

FisherModel::~FisherModel()
{
    if (_descriptors != NULL) 
        delete []_descriptors;

    if (_gmm!=NULL)
        vl_gmm_delete(_gmm);

    if (_means != NULL)
        delete []_means;

    if (_covariances != NULL)
        delete []_covariances;

    if (_priors != NULL)
        delete []_priors;
}
