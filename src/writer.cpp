/*************************************************************************
	> File Name: main.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Thu 30 Jun 2016 04:57:28 PM CST
 ************************************************************************/

#include<iostream>
#include "FsHelpers.h"
#include "scope_timer.hpp"

#include "recog.hpp"
#include "utilities.h"
#include "code_book.h"
#include "fisher_model.h"
#include "kmeans.h"
#include "def.h"

using namespace std;

int TEST_CASIA_My( const char* model_path, const char* gallery_dir,const  char* result_dir )
{
    cv::Size size(60,70);
            
    HfrzRetrival retrival(size, 0, L2_DIST, 2, 1); //fisher + cosine 参数三：1->llc  2->fisher vector
    if (retrival.load_model(model_path, 8) != 0){
        std::cout<<"load model failed !"<<std::endl;
        return -1;
    }

  //std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test_resise_paded_aug";
  //std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test_resise_paded";
  //std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test";
  //trian_codebook(retrival, size, "llc", trian_data, model_path);
  //trian_codebook(retrival, size, "fisher_vector", trian_data, model_path);
  //retrival.load_model(model_path, 8);
  //retrival.get_standard_deviation_with_sub_dir(trian_data, model_path+"/standard_deviation.dat");
  //retrival.load_standard_deviation(model_path+"/standard_deviation.dat");
  bool use_pca = 0;
  uint64_t get_imdb_time = 0; {
      string proc_desc("Get imdb from gallery");
      ScopeTimer scope_time(proc_desc.c_str(), &get_imdb_time);
      std::string pic_dir = "";
      //std::string gallery_dir = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/gallery_resized";
      //std::string gallery_dir = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/gallery";
      retrival.get_imdb_with_sub_dir(gallery_dir, pic_dir, FsHelpers::CombinePath( result_dir,"QinDao.dat") ) ;
  }
  //std::cout << retrival.load_imdb("QinDao.dat") << std::endl;

  //std::string test_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/probe_resized";
  //std::string test_data = "CASIA_probe";
  //vector<float> cmc;
 // std::string result_dir =  "./models_writer_CASIA";
 // int max_size = 10000;

    //CMC_result_sub_dir(retrival, test_data, cmc, result_dir);
 // std::string dst = "metric_train_writer";
 // int test_size = 10;
 // float pca_ratio = 0.98;
  //save_feature_trian_metic( retrival,  src,  dst, test_size, pca_ratio);
  return 0;
}
