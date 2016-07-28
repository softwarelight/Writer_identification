#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

#include <vector>
#include "code_book.h"
#include "FsHelpers.h"

Codebook::Codebook()
{
    _codebookSize = 1024;
    _FeatureDim = 100;
    _patch_num = 0;
    _descriptors = NULL;
    _pCodebook = NULL;
}

int Codebook::get_features(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst)
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
    for (size_t j=0; j<_patch_num; j++) {
        std::cout<<j<<std::endl;
        cv::Mat img = cv::imread(files[j]);
        int channel = img.channels();

        feature.clear();
        retrival._recog.imrecog(img, feature); 
        if (j == 0) {
            feature_size = feature.size();
            fout.write((char*)&feature_size, sizeof(int)); 
            _descriptors = (float*)malloc(_patch_num*feature_size*sizeof(float));
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

int Codebook::load_features(std::string src)
{
    std::ifstream ifp(src.c_str(), std::ios::binary);
    ifp.read((char *)&_patch_num, sizeof(float));
    ifp.read((char *)&_FeatureDim, sizeof(float));
    _descriptors = (float*)malloc(_patch_num*_FeatureDim*sizeof(float));
    _patch_num = 5000;
    for (size_t i = 0; i < _patch_num; i++) {
        int image_class;       
        ifp.read((char*)&image_class, sizeof(int));
        ifp.read((char*)(_descriptors + i*_FeatureDim), sizeof(float)*_FeatureDim);
    }
    return SUCCESS;
}

int Codebook::computeCodebook()
{
    _pCodebook = NULL;
    _pCodebook = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
    if (_pCodebook == NULL)
        return -1;

    (void) vl_kmeans_cluster(_pCodebook, _descriptors, _FeatureDim, _patch_num, _codebookSize);
    return 0;
}

bool Codebook::saveCodebook(const std::string codebookFile)
{
    std::ofstream codebookFs(codebookFile);
    if (!codebookFs.is_open())
    {
        printf("Failed to load codebook file %s\n", codebookFile.c_str());
        return false;
    }
    codebookFs << _pCodebook->numCenters << " " << _pCodebook->dimension << std::endl;
    float *center = (float*)_pCodebook->centers;
    for (int i = 0; i < _pCodebook->numCenters; ++i)
    {
        for (int j = 0; j < _pCodebook->dimension; ++j)
        {
            codebookFs << center[i * _pCodebook->dimension + j] << " ";
        }
        codebookFs << std::endl;
    }
    codebookFs.close();

    return true;
}

bool Codebook::loadCodebook(const std::string codebookFile, VlKMeans** pc)
{
     if (*pc!=NULL)
        vl_kmeans_delete(*pc);
    *pc = NULL;
    *pc = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
    std::ifstream codebookFs(codebookFile);
    if (!codebookFs.is_open())
    {
        printf("Failed to load codebook file %s\n", codebookFile.c_str());
        return false;
    }
    codebookFs >> (*pc)->numCenters >> (*pc)->dimension;
    (*pc)->centers = (float*)malloc((*pc)->numCenters * (*pc)->dimension * sizeof(float*));
    float *data = (float*)(*pc)->centers;
    float value;
    for (int i = 0; i < (*pc)->numCenters; ++i)
    {
        for (int j = 0; j < (*pc)->dimension; ++j)
        {
            codebookFs >> value;
            data[i * (*pc)->dimension + j] = (float)value;
        }
    }
    codebookFs.close();
    return true;
}

bool Codebook::codeCodebook(float *descriptors, int descriptorCount, vl_uint32 *puiAssignments, float *norm)
{
    if ((_pCodebook == NULL) || (descriptors == NULL) || (puiAssignments == NULL) || (descriptorCount < 1))
    {
        return false;
    }

    (void) memset (puiAssignments, 0, descriptorCount * sizeof(vl_uint32));

    int dim = _pCodebook->dimension;
    float *pcodebook = (float*)_pCodebook->centers;

    for (int ii = 0; ii < descriptorCount; ++ii)
    {
        float * feature = descriptors + ii*_pCodebook->dimension;
        float nearest = -10000.0;
        int idx;

        for (size_t jj=0; jj<_pCodebook->numCenters; jj++)
        {
            //__m128 tempsse = _mm_set_ps(0.0f,0.0f,0.0f,0.0f);
            float dist = 0;
            float temp = -norm[jj];

            for (size_t kk = 0; kk < dim; kk++ /*kk+=4*/)
            {
                //tempsse = _mm_add_ps(tempsse, _mm_mul_ps(_mm_loadu_ps(feature+kk), _mm_loadu_ps(pcodebook+dim*jj+kk)));
                dist += *(feature+kk) * (*(pcodebook+dim*jj+kk));
            }
            //temp = temp + tempsse.m128_f32[0] + tempsse.m128_f32[1] + tempsse.m128_f32[2] + tempsse.m128_f32[3];
            temp += dist; 
            if (temp > nearest)
            {
                nearest = temp;
                idx = jj;
            }
        }

        puiAssignments[ii] = idx;
    }

    return true;
}

VlKMeans* Codebook::get_codebook()
{
    return _pCodebook;
}

Codebook::~Codebook()
{
    if (_descriptors != NULL) 
        delete []_descriptors;
    if (_pCodebook!=NULL)
        vl_kmeans_delete(_pCodebook);
}
