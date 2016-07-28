#include <fstream>

#include "FsHelpers.h"
#include "scope_timer.hpp"

#include "recog.hpp"
#include "utilities.h"
#include "code_book.h"
#include "fisher_model.h"
#include "kmeans.h"
#include "def.h"

void save_result(cv::Mat img,const Imdb& image_db, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class);

int get_imdb(HfrzRetrival &retrival, std::string src, std::string dst);
void evaluation(HfrzRetrival& retrival,  int rank, vector<float> &soft, vector<float> &hard, vector<float> &english,\
        vector<float>& greek, std::string result_dir);

int get_imdb_smallpatch(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst);
void test_writer(HfrzRetrival &retrival,  std::string test_set, std::string result, cv::Size size);
void trian_codebook(HfrzRetrival &retrival, cv::Size size, std::string method,std::string trian_data, std::string dst);

int TEST_ICDAR()
{
    std::string model_path = "models_writer";

    //cv::Size size(64,64);
    //cv::Size size(48,48);
    //cv::Size size(32,38);
    cv::Size size(48,58);
    //HfrzRetrival retrival(size, 1, 0, 2, 1); //fisher + pca
    //HfrzRetrival retrival(size, 0, 0, 2, 1); //fisher + L2 //对于cnn做不做pca 要看有没有fisher_model文件
    HfrzRetrival retrival(size, 0, 1, 2, 1); //fisher + cosine
    //HfrzRetrival retrival(size, 0, 0, 1, 1); //llc + L2
    //HfrzRetrival retrival(size, 1, 0, 1, 1); //llc + pca
    if (retrival.load_model(model_path, 8) != 0){
        std::cout<<"load model failed !"<<std::endl;
       // return -1;
    }

    //std::string trian_data = "E:/Write_identification/dataset/icdar2013/verticalCut_resized";
    std::string trian_data = "E:/Write_identification/dataset/icdar2013/verticalCut_paded";
    trian_codebook(retrival, size, "fisher_vector", trian_data, model_path);
    retrival.load_model(model_path, 8);

    //test_writer(retrival, "result_writer", size);
    std::string src = "E:/Write_identification/dataset/icdar2013/experimental_dataset_2013_split";
    std::string dst = "metric_train_writer";
    int test_size = 10;
    float pca_ratio = 0.98;
    //save_feature_trian_metic( retrival,  src,  dst, test_size, pca_ratio);
    return 0;
}
std::string trian_data = "E:/Write_identification/dataset/cvl/cvl-database-1-1/cut_testset_resized_aug_split";
int TEST_CVL()
{
    std::string model_path = "models_writer_CASIA";

    cv::Size size(47,64);
    HfrzRetrival retrival(size, 0, 1, 2, 1); //fisher + cosine

    if (retrival.load_model(model_path, 8) != 0){
        std::cout<<"load model failed !"<<std::endl;
       // return -1;
    }

    std::string trian_data = "E:\\Write_identification\\dataset\\cvl\\cvl-database-1-1\\cut_trainset_resized_aug";
    trian_codebook(retrival, size, "fisher_vector", trian_data, model_path);
    retrival.load_model(model_path, 8);
     std::string test_set =  "E:/Write_identification/dataset/cvl/cvl-database-1-1/cut_testset_resized_aug_split";
    test_writer(retrival,  test_set, "models_writer_CASIA" ,size);
    std::string dst = "metric_train_writer";
    int test_size = 10;
    float pca_ratio = 0.98;
    //save_feature_trian_metic( retrival,  src,  dst, test_size, pca_ratio);
    return 0;
}


int TEST_CASIA()
{
    std::string model_path = "./models_writer_CASIA";

    cv::Size size(60,70);

    HfrzRetrival retrival(size, 0, L2_DIST, 2, 1); //fisher + cosine 参数三：1->llc  2->fisher vector
    if (retrival.load_model(model_path, 8) != 0){
        std::cout<<"load model failed !"<<std::endl;
        // return -1;
    }

   //std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test_resise_paded_aug";
   std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test_resise_paded";
    //std::string trian_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/merged/test";
    //trian_codebook(retrival, size, "llc", trian_data, model_path);
    /*trian_codebook(retrival, size, "fisher_vector", trian_data, model_path);
    retrival.load_model(model_path, 8);
*/
    //retrival.get_standard_deviation_with_sub_dir(trian_data, model_path+"/standard_deviation.dat");
    //retrival.load_standard_deviation(model_path+"/standard_deviation.dat");
    bool use_pca = 0;
    uint64_t get_imdb_time = 0; {
        string proc_desc("Get imdb from gallery");
        ScopeTimer scope_time(proc_desc.c_str(), &get_imdb_time);
        std::string pic_dir = "";
        std::string gallery_dir = "./CASIA_gallery";
       //std::string gallery_dir = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/gallery_resized";
        //std::string gallery_dir = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/gallery";
        retrival.get_imdb_with_sub_dir(gallery_dir, pic_dir, "QinDao.dat");
    }
    std::cout << retrival.load_imdb("QinDao.dat") << std::endl;

    std::string test_data = "./CASIA_probe";
    //std::string test_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/probe_resized";
    //std::string test_data = "E:/Write_identification/dataset/CASIA-HWDB2.1/probe_and_gallery_train/probe";
    vector<float> cmc;
    std::string result_dir =  "./models_writer_CASIA";
    int max_size = 10000;

    CMC_result_sub_dir(retrival, test_data, cmc, result_dir);
    std::string dst = "metric_train_writer";
    int test_size = 10;
    float pca_ratio = 0.98;
    //save_feature_trian_metic( retrival,  src,  dst, test_size, pca_ratio);
    return 0;
}

void trian_codebook(HfrzRetrival &retrival, cv::Size size, std::string method,std::string trian_data, std::string dst)
{

    //std::string trian_data = "E:/Write_identification/dataset/icdar2013/trian_crop_small_patch";
    //std::string trian_data = "E:/Write_identification/dataset/icdar2013/trian_crop_small_patch_64_64";
    std::string feature_data = "vertical_feature_data.dat";
    if(method == "llc") {
         Codebook cb;
         cb.get_features(retrival, size, trian_data, feature_data);
         //cb.load_features(feature_data);
         cb.computeCodebook();
         std::string dst1 = FsHelpers::CombinePath(dst, "codebook_llc.dat");
         cb.saveCodebook(dst1);
    } else if(method == "fisher_vector") {
        FisherModel fm(256);
        fm.get_features(retrival, size, trian_data, feature_data);
        //fm.load_features(feature_data);
        float pca_ratio = 0.995;
        fm.computeCodebook(pca_ratio, FsHelpers::CombinePath(dst, "fisher_pca.xml"));
        std::string dst1 = FsHelpers::CombinePath(dst, "gmm_model.dat");
        fm.saveCodebook(dst1);
    } else {
        std::cout<< "input method error"<<std::endl;
    }
}

void test_writer(HfrzRetrival &retrival, std::string test_set, std::string result, cv::Size size)
{

     //std::string gallery_dir = "E:\\Write_identification\\dataset\\icdar2013\\icdar2013_benchmarking_dataset_RGB_1";
     //std::string gallery_dir = "E:\\Write_identification\\dataset\\icdar2013\\icdar2013_benchmarking_verticalCut_resized";
     //std::string gallery_dir = "E:\\Write_identification\\dataset\\icdar2013\\icdar2013_benchmarking_verticalCut_paded";
    std::string gallery_dir = test_set;
     bool use_pca = 1;
     uint64_t get_imdb_time = 0; {
         string proc_desc("Get imdb from gallery");
         ScopeTimer scope_time(proc_desc.c_str(), &get_imdb_time);
         std::string pic_dir = "E:\\Write_identification\\dataset\\icdar2013\\icdar2013_benchmarking_dataset_RGB_1";
         retrival.get_imdb_with_sub_dir(gallery_dir, pic_dir, "2imdb_writer_fisher_995_512k_norm.dat");
     }

     std::cout << retrival.load_imdb("2imdb_writer_fisher_995_512k_norm.dat") << std::endl;

     vector<float> soft;
     vector<float> hard;
     vector<float> english;
     vector<float> greek;

     std::string dst = result;
     int rank = 500;
     evaluation(retrival, rank, soft, hard, english, greek, "result_writer\\pics\\");

     std::ofstream fresult(dst + "/result_writer.txt");
     std::ofstream fsoft(dst + "/result_writer_soft.txt");
     std::ofstream fhard(dst + "/result_writer_hard.txt");
     std::ofstream fenglish(dst + "/result_writer_english.txt");
     std::ofstream fgreek(dst + "/result_writer_greek.txt");

     std::cout<<std::endl<< "soft evaluation result: "<<std::endl;
     fresult<<std::endl<< "soft evaluation result: "<<std::endl;
     for (int i =0; i <soft.size(); i++){
        fsoft<<i<<" : "<<soft[i]<<std::endl;
        if(i <=10) {
            std::cout<<i<<":"<<soft[i]<<std::endl;
            fresult<<i<<" : "<<soft[i]<<std::endl;
        }
     }
     fsoft.close();

     std::cout<<std::endl<< "hard evaluation result: "<<std::endl;
     fresult<<std::endl<< "hard evaluation result: "<<std::endl;
     for (int i = 0; i < hard.size(); i++){
        fhard<<i<<" : "<<hard[i]<<std::endl;
        if(i <=10) {
            std::cout<<i<<":"<<hard[i]<<std::endl;
            fresult<<i<<" : "<<hard[i]<<std::endl;
        }
     }
     fhard.close();

     std::cout<<std::endl<< "english evaluation result: "<<std::endl;
     fresult<<std::endl<< "english evaluation result: "<<std::endl;
     for (int i = 0; i < english.size(); i++){
        fenglish<<i<<" : "<<english[i]<<std::endl;
        if(i <=10) {
            std::cout<<i<<":"<<english[i]<<std::endl;
            fresult<<i<<" : "<<english[i]<<std::endl;
        }
     }
     fenglish.close();

     std::cout<<std::endl<< "greek evaluation result: "<<std::endl;
     fresult<<std::endl<< "greek evaluation result: "<<std::endl;
     for (int i = 0; i < greek.size(); i++){
        fgreek<<i<<" : "<<greek[i]<<std::endl;
        if(i <=10) {
            std::cout<<i<<":"<<greek[i]<<std::endl;
            fresult<<i<<" : "<<greek[i]<<std::endl;
        }
     }
     fgreek.close();
     fresult.close();
}

bool cmp(std::pair<int,float> a, std::pair<int,float> b)
{
    return a.second>b.second;
}
void evaluation(HfrzRetrival& retrival,  int rank, vector<float> &soft, vector<float> &hard, vector<float> &english,\
    vector<float>& greek, std::string result_dir)
{
    if(!FsHelpers::Exists(result_dir))
        FsHelpers::MakeDirectories(result_dir);

    std::vector<std::pair<int,float> > top_result;
    int flag;
    const Imdb &image_db = retrival.get_imdb();
    int pic_size = image_db.size;
    int dim = image_db.dimension;
    for (int i = 0; i < pic_size; i++) {
        std::cout<<"processing "<<i<<std::endl;

        int img_class = image_db.image_class[i];
        cv::Mat img = cv::imread(image_db.image_path[i]);
        string img_name = FsHelpers::GetFileName(image_db.image_path[i]);

        flag = 0;
        top_result.clear();
        for(int j = 0; j<pic_size; j++){
            if(j == i)      continue;

            float dist;
            if (retrival._dist_method == METRIC_DIST)
            {
                dist =retrival.get_distance_metric(&image_db.imdb[i][0], &image_db.imdb[j][0], dim);
            } else if (retrival._dist_method == L2_DIST) {
                dist = retrival.get_distance(&image_db.imdb[i][0], &image_db.imdb[j][0], dim);
            } else if (retrival._dist_method == COSINE_DIST) {
                dist = retrival.get_distance_cosine(&image_db.imdb[i][0], &image_db.imdb[j][0], dim);
            }
            top_result.push_back(std::make_pair(j,dist));
        }

        if (retrival._dist_method != COSINE_DIST) {
            for (size_t j = 0; j < top_result.size(); j++) {
                top_result[j].second = top_result[j].second + 0.00001;
                top_result[j].second = 1/top_result[j].second;
            }
        }
        sort(top_result.begin(), top_result.end(), cmp);


        int hard_end;
        int m;
#ifdef TEST_ICDAR
        save_result(img, image_db, top_result, img.size(), i, result_dir,img_class);
        hard_end = 3;
        m = 1000;
#endif
#ifdef TEST_CVL
        hard_end = 5;
        m = 10000;
#endif
        soft.resize(rank+1,0);
        hard.resize(hard_end+1,0);

#ifdef TEST_ICDAR
    english.resize(2001,0);
    greek.resize(2001,0);
#endif
        for (int j = 1; j<=hard_end; j++){
            int idx = top_result[j-1].first;
            if (image_db.image_class[idx] % m == img_class % m)
                hard[j]++;
            else
                break;
        }

        for (int j = 1; j <= rank; j++) {
            if (flag) {
                soft[j]++;
            } else {
                int idx = top_result[j-1].first;
                if (image_db.image_class[idx] % m == img_class % m) {
                    soft[j]++;
                    flag = 1;
                }
            }
        }

#ifdef TEST_ICDAR
        flag = 0;
        int cnt = 0;
        if(img_class/1000 == 1  || img_class/1000 ==2 )
            for(int j = 0; j<top_result.size(); j++) {
                int idx = top_result[j].first;
                int samlp_class = image_db.image_class[idx];
                if(samlp_class / 1000 == 1 || samlp_class /1000 == 2) {
                    cnt++;
                    if(flag)
                        english[cnt]++;
                    else if (samlp_class % 1000 == img_class % 1000) {
                        english[cnt]++;
                        flag = 1;
                    }
                }
            }

        flag = 0;
        cnt = 0;
        if(img_class/1000 == 3  || img_class/1000 ==4 )
            for(int j = 0; j<top_result.size(); j++) {
                int idx = top_result[j].first;
                int samlp_class = image_db.image_class[idx];
                if(samlp_class / 1000 == 3 || samlp_class /1000 == 4) {
                    cnt++;
                    if(flag)
                        greek[cnt]++;
                    else if (samlp_class % 1000 == img_class % 1000) {
                        greek[cnt]++;
                        flag = 1;
                    }
                }
            }
#endif
    }
    std::fstream fout("log_time.txt", std::ios::app);

    double tv_ms = extract_feature_time*1.0/pic_size/1000;
    fout<<"extract_feature_time : "<<tv_ms<<"(ms)"<<std::endl;

    tv_ms = query_image_time_once*1.0/pic_size/1000;
    fout<<"query_image_time_once : "<<tv_ms<<"(ms)"<<std::endl;

    fout.close();

    for (int i=1; i <= rank; i++) {
        soft[i] /= pic_size;
    }

    for (int i=1; i < hard.size(); i++) {
        hard[i] /= pic_size;
    }

#ifdef TEST_ICDAR
    for (int i=1; i < english.size(); i++) {
        english[i] /= (pic_size/2);
    }

    for (int i=1; i < greek.size(); i++) {
        greek[i] /= (pic_size/2);
    }
#endif
}



void save_result(cv::Mat img,const Imdb& image_db, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class)
{
    int num=6;
    /*cv::Mat dst_img(image_size.height*num+(num-1)*4, image_size.width*num+(num-1)*4, img.type(), cv::Scalar(0,255,0));
    char dst_image_name[20];
    itoa(test_index, dst_image_name, 10);

    cv::Rect dst_roi_s(0, 0, image_size.width, image_size.height);
    cv::Mat img_roi_s = dst_img(dst_roi_s);
    img.copyTo(img_roi_s);

    CvxText text("simhei.ttf");
    float p = 1;
    CvScalar size = cvScalar(600, 0.1, 0.1, 0);

	text.setFont(NULL, &size, NULL, &p);   // 透明处理

    for (int i = 1; i<num*num; i++) {
        int idx = top_result[i-1].first;
        cv::Mat img_temp = cv::imread(image_db.image_path[idx]);
        cv::Mat img_resized(img.size(), img.type());
        cv::resize(img_temp, img_resized, img_resized.size());

        cv::Rect dst_roi((i%num)*image_size.width+(i%num)*4, (i/num)*image_size.height+(i/num)*4, image_size.width, image_size.height);
        cv::Mat img_roi = dst_img(dst_roi);

        img_resized.copyTo(img_roi);
        if (image_db.image_class[idx]%1000 == img_class%1000) {
            string msg("Y");
            text.putText(&IplImage(img_roi), msg.c_str(), cvPoint(5, 400), CV_RGB(255,0,0));
        }
    }
    cv::imwrite(FsHelpers::CombinePath(result_dir, dst_image_name)+".jpg", dst_img);
    */
}



int get_imdb_smallpatch(HfrzRetrival &retrival, cv::Size patch_size, std::string src, std::string dst)
{
    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    }

    fout.seekp(fout.beg + 4);
    int sz = 0;

    vector<string> files;
    FsHelpers::GetFilesHasExtension(src, files, ".jpg",false);

    int width = patch_size.width;
    int height = patch_size.height;

    sz += files.size();
    cv::Size img_sz(2500, 500);
    for (size_t j=0; j<files.size(); j++) {
        std::cout<<j<<std::endl;
        cv::Mat img = cv::imread(files[j]);
        cv::Mat img_resized;
        cv::resize(img,img_resized,img_sz,0,0, cv::INTER_CUBIC);

        int channel = img.channels();
        std::vector<float> features;
        for(int r = 0; r <= 500 - height; r = r + height/4) {
            for(int c = 0; c <= 2300 - width; c += width*10) {
                vector<float> feature;
                cv::Rect rect(c, r, width, height);
                cv::Mat img_roi = img_resized(rect);
                retrival._recog.imrecog(img_roi,feature);
                features.insert(features.end(), feature.begin(), feature.begin() + 100);
            }
        }
        if (j == 0) {
            int feature_size = features.size();
            fout.write((char*)&feature_size, sizeof(int));
        }

        std::string name = FsHelpers::GetFileName(files[j]);
        int id1 = atoi(name.substr(0,3).c_str());
        int id2 = (name[4] - '0')*1000;
        int id = id1 + id2;

        int path_size = files[j].size();
        const char *path = files[j].c_str();
        fout.write((char*)&path_size, sizeof(int));
        fout.write(path, sizeof(char)*(path_size+1));
        fout.write((char*)&id, sizeof(int));
        fout.write((char*)&features[0], sizeof(float) * features.size());
    }


    fout.seekp(fout.beg);
    fout.write((char*)&sz,sizeof(sz));

    fout.close();
    return 0;
}

int get_imdb(HfrzRetrival &retrival, std::string src, std::string dst)
{
    std::ofstream fout(dst.c_str(), std::ios::binary | std::ios::out);
    if (!fout.is_open()) {
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    }

    fout.seekp(fout.beg + 4);
    int sz = 0;

    vector<string> files;
    FsHelpers::GetFilesHasExtension(src, files, ".jpg",false);

    sz += files.size();
    cv::Size img_sz(2500, 500);
    for (size_t j=0; j<files.size(); j++) {
        std::cout<<j<<std::endl;
        cv::Mat img = cv::imread(files[j]);
        cv::Mat img_resized;
        cv::resize(img,img_resized,img_sz,0,0, cv::INTER_CUBIC);

        int channel = img.channels();
        std::vector<float> features;
        for(int i = 0; i<=2000; i+=100) {
            vector<float> feature;
            cv::Rect rect(i, 0, 500,500);
            cv::Mat img_roi = img_resized(rect);
            retrival._recog.imrecog(img_roi,feature);
            features.insert(features.end(), feature.begin(), feature.begin() + 100);
        }

        if (j == 0) {
            int feature_size = features.size();
            fout.write((char*)&feature_size, sizeof(int));
        }

        std::string name = FsHelpers::GetFileName(files[j]);
        int id1 = atoi(name.substr(0,3).c_str());
        int id2 = (name[4] - '0')*1000;
        int id = id1 + id2;

        int path_size = files[j].size();
        const char *path = files[j].c_str();
        fout.write((char*)&path_size, sizeof(int));
        fout.write(path, sizeof(char)*(path_size+1));
        fout.write((char*)&id, sizeof(int));
        fout.write((char*)&features[0], sizeof(float) * features.size());
    }


    fout.seekp(fout.beg);
    fout.write((char*)&sz,sizeof(sz));

    fout.close();
    return 0;
}
