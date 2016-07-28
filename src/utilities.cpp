#include "utilities.h"
#include "result_sort.h"
#include "def.h"
#include "scope_timer.hpp"
#include"FsHelpers.h"

using namespace cv;
struct IdNum{
    int id;
    int sub_size;    
};
int save_feature_trian_metic(HfrzRetrival& retrival, const std::string src, const std::string dst, \
    const int test_size, const float pca_ratio)
{
    if (!FsHelpers::Exists(dst)) FsHelpers::MakeDirectory(dst);

    std::vector<std::string> directories;
    FsHelpers::GetDirectories(src,directories);

    std::ofstream f_feature(FsHelpers::CombinePath(dst,"features_pca.txt").c_str());
    std::ofstream f_idxa(FsHelpers::CombinePath(dst,"idxa.txt").c_str());
    std::ofstream f_idxb(FsHelpers::CombinePath(dst,"idxb.txt").c_str());
    std::ofstream f_matches(FsHelpers::CombinePath(dst,"matches.txt").c_str());

    if (!f_feature.is_open() || !f_idxa.is_open() || !f_idxb.is_open() || !f_matches.is_open()) {
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    }

    // 假定库里面的图片总数目不超过file_sz, file_sz一定要大于总图数目
    int file_sz = directories.size() * 40; 
    int count = 0;

    std::cout << "directories size:" << directories.size() << std::endl;
    int feature_sz = 0;
    srand(1);

    cv::Mat features;


    vector<IdNum> class_nums;

    for (size_t i = 0; i < directories.size();  i++) {     

        vector<string> files;
        FsHelpers::GetFilesHasExtension(directories[i], files, ".jpg",false);
        int sz = files.size();    
        std::cout << i << std::endl;

        //每一个子类最多取200图
        int sub_size = i < test_size ? 2 : (sz < 200 ? sz : 200);

        IdNum temp;
        temp.id = count;
        temp.sub_size = sub_size;
        class_nums.push_back(temp);

        int a, b;
        for (size_t j=0; j<sub_size ; j++) {           
            cv::Mat img; 

            if (i < test_size) {
                int r = rand();
                if (j == 0) {
                    a = r % sz;
                    img = cv::imread(files[a]);
                } else {                
                    b = r % sz;
                    while (a == b) {                    
                        b = rand() % sz;                       
                    } 
                    img = cv::imread(files[b]);
                }

            } else {            
                img = cv::imread(files[j]);
            }   

            std::vector<float> feature;
            if(!retrival._code_method) {
                cv::Mat im_resized(retrival.get_recog_size(), img.type());
                cv::resize(img, im_resized, im_resized.size(),0,0, CV_INTER_CUBIC);
                std::cout<<j<<":"<<retrival._recog.imrecog(im_resized,feature)<<std::endl; 
            } else {
                std::cout<<j<<":"<<retrival._fea_handler->extract_feature(img, feature, retrival._llc_step)<<";"<<i<<std::endl;
            }

            if (i==0 && j==0){                       
                features.create(file_sz, feature.size(), CV_32FC1);
            }
                
            for (size_t k = 0; k < feature.size(); k++){
                features.at<float>(count,k) = feature[k];   
            } 
            count++;
        }           
    }       

    cv::Mat rfeatures = features(cv::Rect(0, 0, features.cols, count));
    cv::Mat pca_features;
    if(pca_ratio) { 
        std::string pca_dst = FsHelpers::CombinePath(dst,"pca_config.xml");
        mat_pca(rfeatures, pca_dst, pca_ratio, pca_features);
    } else {
        pca_features = rfeatures;
    }  

    for (size_t i = 0; i < pca_features.rows; i++) {    
        for (size_t j = 0; j < pca_features.cols; j++) {        
            f_feature << pca_features.at<float>(i,j)<<" ";
        }
        f_feature << std::endl;
    }

    for (size_t i = 0; i < test_size; i++) {    
        f_idxa << i * 2 + 1 << " ";
        f_idxb << i * 2 + 2 << " ";
        f_matches << "1" << " ";
    }

    //generate idxa and idxb  matches
    for (size_t i = test_size; i < class_nums.size(); i++) {
        int sz = class_nums[i].sub_size;
        int id = class_nums[i].id + 1; //because matlab coordinate begin with 1

        //for car and window and  people
       /* for (size_t j = 0; j < sz; j++) {
            for (size_t k = 0; k < 3; k++) {            
                int ida = id + j;
                int idb = id + (j + k*3) % sz;
                if (ida != idb) {                
                    f_idxa << ida << " ";
                    f_idxb << idb << " ";
                    f_matches << "1" << " ";
                }
            }
        }*/

         for (size_t j = 0; j < sz; j++) {
            for (size_t k = j+1; k < sz; k++) {            
                int ida = id + j;
                int idb = id + k;
                if (ida != idb) {                
                    f_idxa << ida << " ";
                    f_idxb << idb << " ";
                    f_matches << "1" << " ";
                }
            }
        }
    }

    //generate idxa and idxb don't matches
    for (size_t i = test_size; i < class_nums.size(); i++) {    
        int sz = class_nums[i].sub_size;
        int id = class_nums[i].id + 1; //because matlab coordinate begin with 1
        for (size_t j = 0; j < sz; j++) {        
            for (size_t k = 0; k < 3; k++) {            
                int ida = id + j;
               
                int b ;
                do {                
                    b = rand() % class_nums.size();
                } while (b == i || b < test_size);

                int idb = class_nums[b].id + 1 + (rand() % class_nums[b].sub_size);
                if (ida != idb) { //just for protect                
                    f_idxa << ida << " ";
                    f_idxb << idb << " ";
                    f_matches << "0" << " ";
                }
            }
        }
    }

    f_feature.close();  
    f_idxa.close();
    f_idxb.close();
    f_matches.close();

    return 0;
}

int save_feature_simple(const std::string src, const std::string dst, const std::string model_path, const float pca_ratio)
{
    hfrecog recog;
    recog.create(FsHelpers::CombinePath(model_path, "md.dat"), PASS_WORD, 0);
    std::vector<std::string> directories;
    FsHelpers::GetDirectories(src,directories);

    std::ofstream fout(dst.c_str());
    if (!fout.is_open()) {    
        std::cout<<"the file can't open correctly!"<<std::endl;
        return -1;
    }

    int file_sz = directories.size() * 2;
    std::cout << "directories size:" << directories.size() << std::endl;
    int feature_sz = 0;
    srand(1);

    cv::Mat features;//directories.size()
    for (size_t i = 0; i < directories.size(); i++) {             
        std::cout<<i<<std::endl;
        vector<string> files;
        FsHelpers::GetFilesHasExtension(directories[i], files, ".jpg",false);
        int sz = files.size();

        int a , b;

        for (size_t j=0; j<2; j++) {    
            cv::Mat img;  
            int r = rand();
            if (j == 0) {
                a = r % sz;
                img = cv::imread(files[a]);
            } else {
                b = r % sz;
                while (a == b) {                
                    b = rand() % sz;
                }
                img = cv::imread(files[b]);
            }

            std::vector<float> feature;
            recog.imrecog(img,feature); 

            if (i==0 && j==0) {
                features.create(file_sz, feature.size(), CV_32FC1);
            }

            for (size_t k = 0; k < feature.size(); k++) {
                features.at<float>(i*2+j,k) = feature[k];   
            }    
        }        
    }

    cv::Mat pca_features;
    if (pca_ratio) {
        std::string pca_dst = FsHelpers::CombinePath(dst,"pca_config.xml");
        mat_pca(features, pca_dst, pca_ratio, pca_features);
    } else {
        pca_features = features;
    }

    for (size_t i = 0; i < pca_features.rows; i++) {
        for (size_t j = 0; j < pca_features.cols; j++) {
            fout << pca_features.at<float>(i,j)<<" ";
        }
        fout << std::endl;
    }
    fout.close();  
    return 0;
}

int mat_pca(const cv::Mat &indata, const std::string &dst, const float pca_ratio, cv::Mat &outdata)
{
    int dementions = indata.cols;  
    int rows = indata.rows;
    
    //Training
    cv::PCA *pca = new cv::PCA(indata, cv::Mat(), CV_PCA_DATA_AS_ROW);
   
    //calculate the decreased dimensions
    int index;
    float sum=0, sum0=0, ratio;
    for (int d=0; d<pca->eigenvalues.rows; ++d) {
        sum += pca->eigenvalues.at<float>(d,0);
    }

    for (int d=0; d<pca->eigenvalues.rows; ++d) {
        sum0 += pca->eigenvalues.at<float>(d,0);
        ratio = sum0 / sum;
        if (ratio > pca_ratio) {
            index = d+1;
            break;
        }
    }

    std::cout << "index:" << index << std::endl;
    std::cout << pca->eigenvectors.rows<<"  ";
    std::cout << pca->eigenvectors.cols;
    pca->eigenvectors = pca->eigenvectors(cv::Rect(0,0,dementions,index));
    Mat eigenvetors_d = pca->eigenvectors;

    //write mean and eigenvalues into xml file
    cv::FileStorage fs_w(dst, FileStorage::WRITE);
    fs_w << PCA_MEAN << pca->mean;
    fs_w << PCA_EIGEN_VECTOR << eigenvetors_d;
    fs_w << PCA_EIGEN_VALUE << pca->eigenvalues;
    fs_w.release();

    //Encoding
    outdata.create(rows, index, CV_32FC1);    
    pca->project(indata, outdata);
    std::cout << std::endl << "pca_encode dimensions:" << std::endl << index << std::endl;
   
    delete pca;
    return 0;
}

void save_result_image(cv::Mat img,const Imdb& image_db, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class)
{
    int num=6;
    /*Mat dst_img(image_size.height*num+(num-1)*4, image_size.width*num+(num-1)*4, img.type(), cv::Scalar(0,255,0));
    char dst_image_name[20];
    itoa(test_index, dst_image_name, 10);

    cv::Rect dst_roi_s(0, 0, image_size.width, image_size.height);
    cv::Mat img_roi_s = dst_img(dst_roi_s);
    img.copyTo(img_roi_s);

    CvxText text("simhei.ttf");
    float p = 1;
    CvScalar size = cvScalar(45, 0.1, 0.1, 0);

	text.setFont(NULL, &size, NULL, &p);   // 透明处理

    for (int i = 1; i<num*num; i++) {
        int idx = top_result[i-1].first;
        cv::Mat img_temp = cv::imread(image_db.image_path[idx]);
        cv::Mat img_resized(img.size(), img.type());
        cv::resize(img_temp, img_resized, img_resized.size());

        cv::Rect dst_roi((i%num)*image_size.width+(i%num)*4, (i/num)*image_size.height+(i/num)*4, image_size.width, image_size.height);
        cv::Mat img_roi = dst_img(dst_roi);    

        img_resized.copyTo(img_roi);
        if (image_db.image_class[idx] == img_class) {
            string msg("Y");
            text.putText(&IplImage(img_roi), msg.c_str(), cvPoint(5, 5), CV_RGB(255,0,0));
        }
    }
    cv::imwrite(FsHelpers::CombinePath(result_dir, dst_image_name)+".jpg", dst_img);*/
}

void save_result_image_two(cv::Mat img,vector<Mat> & imdb, vector<int> imdb_class, std::vector<std::pair<int,float> > &top_result,cv::Size image_size, \
    int test_index, std::string result_dir, int img_class)
{
    int num=6;
   /* Mat dst_img(image_size.height*num+(num-1)*4, image_size.width*num+(num-1)*4, img.type(), cv::Scalar(0,255,0));
    char dst_image_name[20];
    itoa(test_index, dst_image_name, 10);

    cv::Rect dst_roi_s(0, 0, image_size.width, image_size.height);
    cv::Mat img_roi_s = dst_img(dst_roi_s);
    img.copyTo(img_roi_s);

    CvxText text("simhei.ttf");
    float p = 1;
    CvScalar size = cvScalar(60, 0.1, 0.1, 0);

    text.setFont(NULL, &size, NULL, &p);   // 透明处理

    for (int i = 1; i<num*num; i++) {
        int idx = top_result[i-1].first;
        cv::Mat img_temp = imdb[idx];
        cv::Mat img_resized(img.size(), img.type());
        cv::resize(img_temp, img_resized, img_resized.size());

        cv::Rect dst_roi((i%num)*image_size.width+(i%num)*4, (i/num)*image_size.height+(i/num)*4, image_size.width, image_size.height);
        cv::Mat img_roi = dst_img(dst_roi);    

        img_resized.copyTo(img_roi);
        if (imdb_class[idx] == img_class) {
            string msg("Y");
            text.putText(&IplImage(img_roi), msg.c_str(), cvPoint(5, 55), CV_RGB(255,0,0));
        }
    }
    cv::imwrite(FsHelpers::CombinePath(result_dir, dst_image_name)+".jpg", dst_img);*/
}

void CMC_result_two(HfrzRetrival& retrival, std::string gallery_dir, std::string probe_dir, int rank, vector<float> &cmc,\
    std::string result_dir, std::string method, int max_size, bool use_pca)
{
    vector<string> gallery_files;
    FsHelpers::GetFilesHasExtension(gallery_dir, gallery_files, ".jpg", true);
    FsHelpers::GetFilesHasExtension(gallery_dir, gallery_files, ".bmp",false);
    int gallery_size = gallery_files.size();
    vector<Mat> imdb(gallery_size, Mat());
    vector<int> imdb_class;
    cv::Size image_size = retrival.get_recog_size();
    std::cout<<"gallery_size： "<<gallery_size<<std::endl;
    for (int i = 0; i < gallery_size; i++) {
        cv::Mat image = imread(gallery_files[i]);

        cv::Mat im_resized(image_size, image.type());  
        if(image_size != image.size())
            cv::resize(image, im_resized, im_resized.size(),0,0,CV_INTER_CUBIC);
        else
            im_resized = image;
        imdb[i] = im_resized.clone();

        string img_name = FsHelpers::GetFileName(gallery_files[i]);
        int cls = atoi((img_name.substr(0,5)).c_str());
        imdb_class.push_back(cls);
    }

    vector<string> probe_files;
    FsHelpers::GetFilesHasExtension(probe_dir, probe_files, ".jpg", true);
    FsHelpers::GetFilesHasExtension(probe_dir, probe_files, ".bmp",false);

    std::vector<float> sum(rank+1,0);
    std::vector<std::pair<int,float>> top_result;
    bool flag;
    int pic_size = probe_files.size() > max_size ? max_size:probe_files.size();

    for (int i = 0; i < pic_size; i++) {
        std::cout<<"processing "<<i<<std::endl;
        string img_name = FsHelpers::GetFileName(probe_files[i]);
        int img_class = atoi((img_name.substr(0,5)).c_str());

        flag = 0;
        top_result.clear();


        cv::Mat img = cv::imread(probe_files[i]);    
        cv::Mat im_resized(image_size, img.type());  
        if(image_size != img.size())
            cv::resize(img, im_resized, im_resized.size(),0,0,CV_INTER_CUBIC);
        else
            im_resized = img;
       
        cv::Mat image_recog(image_size.height, image_size.width*2, img.type());
        cv::Rect roi(0,0,image_size.width, image_size.height);
        cv::Mat image_recog_roi = image_recog(roi);

        im_resized.copyTo(image_recog_roi);
        //cv::imshow("img",image_recog);
       // cv::waitKey();

        vector<float> scores;
        for(int j = 0; j < gallery_size; j++)
        {
            roi.x = image_size.width;
            image_recog_roi = image_recog(roi);
            imdb[j].copyTo(image_recog_roi);
            std::vector<float> score;
            /*cv::imshow("img",image_recog);
            cv::waitKey();*/
            retrival._recog.imrecog(image_recog, score);
            scores.push_back(score[1]);
        } 
        result_sort(scores, rank, top_result);

        save_result_image_two(img, imdb, imdb_class , top_result, img.size(), i, result_dir,img_class);

        flag = 0;
        for (int j = 1; j <= rank; j++) {
            if (flag) {
                sum[j]++;
            } else {
                int idx = top_result[j-1].first;
                if (imdb_class[idx] == img_class) {
                    sum[j]++;
                    flag = 1;
                }
            }			
        }
    }

    std::fstream fout("log_time.txt", std::ios::app);

    double tv_ms = extract_feature_time*1.0/pic_size/1000;
    fout<<"extract_feature_time : "<<tv_ms<<"(ms)"<<std::endl;

    tv_ms = query_image_time_once*1.0/pic_size/1000;
    fout<<"query_image_time_once : "<<tv_ms<<"(ms)"<<std::endl;

    fout.close();

    for (int i=1; i <= rank; i++) {
        sum[i] /= pic_size;
    }
    cmc = sum;
}

void CMC_result(HfrzRetrival& retrival, std::string src, int rank, vector<float> &cmc,\
        std::string result_dir, int max_size, bool use_pca)
{
    vector<string> files;
    FsHelpers::GetFilesHasExtension(src, files, ".jpg", true);
    FsHelpers::GetFilesHasExtension(src, files, ".bmp",false);
    std::vector<float> sum(rank+1,0);
    std::vector<std::pair<int,float> > top_result;
    int flag;
    int pic_size = files.size() > max_size ? max_size:files.size();

    for (int i = 0; i < pic_size; i++) {
        std::cout<<"processing "<<i<<std::endl;
        string img_name = FsHelpers::GetFileName(files[i]);
        int img_class = atoi((img_name.substr(0,5)).c_str());

        flag = 0;
        top_result.clear();

        cv::Mat img = cv::imread(files[i]);
        cv::Mat im_resized(retrival.get_recog_size(), img.type());  
        cv::resize(img, im_resized, im_resized.size(),0,0,CV_INTER_CUBIC);

        uint64_t qurey_time = 0;{
            string proc_desc;
            ScopeTimer scope_time(proc_desc.c_str(), &qurey_time);
            retrival.query_image(im_resized, use_pca, rank, top_result);
        }
        query_image_time_once = query_image_time_once + qurey_time;

        const Imdb &image_db = retrival.get_imdb();

        save_result_image(img, image_db, top_result, img.size(), i, result_dir,img_class);

        for (int j = 1; j <= rank; j++) {
            if (flag) {
                sum[j]++;
            } else {
                int idx = top_result[j-1].first;
                if (image_db.image_class[idx] == img_class) {
                    sum[j]++;
                    flag = 1;
#ifdef HFRZ_DEBUG                
                    std::string pic_dir = FsHelpers::CombinePath(result_dir, img_name.substr(0,5));

                    if (!FsHelpers::Exists(pic_dir)) FsHelpers::MakeDirectories(pic_dir);

                    const Imdb &image_db = retrival.get_imgb();
                    std::string img_path_gallery =image_db.image_path[idx];
                    string img_name_gallery = FsHelpers::GetFileName(img_path_gallery);
                    cv::Mat img_gallery = cv::imread(image_db.image_path[idx]);

                    cv::imwrite(FsHelpers::CombinePath(pic_dir, img_name), img);
                    cv::imwrite(FsHelpers::CombinePath(pic_dir, img_name_gallery), img_gallery);                    
#endif
                }
            }			
        }
    }

    std::fstream fout("log_time.txt", std::ios::app);

    double tv_ms = extract_feature_time*1.0/pic_size/1000;
    fout<<"extract_feature_time : "<<tv_ms<<"(ms)"<<std::endl;

    tv_ms = query_image_time_once*1.0/pic_size/1000;
    fout<<"query_image_time_once : "<<tv_ms<<"(ms)"<<std::endl;

    fout.close();

    for (int i=1; i <= rank; i++) {
        sum[i] /= pic_size;
    }
    cmc = sum;
}

int evaluation( const vector< vector<float> >& distanceMatrix ,const int& rank ) {
    for ( size_t i = 0 ; i < distanceMatrix.size(); i++ ) {
        for ( size_t j = 0; j < distanceMatrix[0].size(); j ++ ) {
            
        }
    }
    return 0;
}

void CMC_result_sub_dir(HfrzRetrival& retrival, std::string src, vector<float> &cmc, std::string dst)
{
    double map = 0;
    if ( ! FsHelpers::Exists( dst ) ) {
        FsHelpers::MakeDirectories( dst );
    }
    
    std::ofstream topResult( dst + "/topK.txt");
    
    const Imdb &image_db = retrival.get_imdb();
    int rank = image_db.imdb.size();
    int N = 30 > rank ? rank : 30;

    vector<string> dirs;
    FsHelpers::GetDirectories(src, dirs);

    std::vector<float> sum(rank+1,0);
    std::vector<std::pair<int,float> > top_result;
    int flag;
   
    int dir_size = dirs.size();
    for (int i = 0; i < dir_size; i++) {
        std::cout<<"processing "<<i<<'\t'<<dirs[i]<<std::endl;
        string img_name = FsHelpers::GetFileName(dirs[i]);
        int img_class = atoi((img_name.substr(0,5)).c_str());

        flag = 0;
        top_result.clear();

        uint64_t qurey_time = 0;{
            string proc_desc;
            ScopeTimer scope_time(proc_desc.c_str(), &qurey_time);
            retrival.query_image_patch(dirs[i], top_result);
        }

////////////////////////////////////////////////////////////////////////////////////////////
topResult<<dirs[i] + ".jpg"<<endl;
topResult<<N<<endl;
for ( int j = 0 ;j < N;j ++ ) {
    int idx = top_result[j].first;
    topResult<< image_db.image_path[ idx ]<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////////

        query_image_time_once = query_image_time_once + qurey_time; 
        //save_result_image(img, image_db, top_result, img.size(), i, result_dir,img_class);

        for (int j = 1; j <= rank; j++) {
            if (flag) {
                sum[j]++;
            } else {
                int idx = top_result[j-1].first;
                if (image_db.image_class[idx] == img_class) {
                    sum[j]++;
                    flag = 1;
                    map += 1.0 / j;
                    //cout<<j<<'\t'<<idx<<'\t'<<image_db.image_class[idx]<<'\t'<<img_class<<endl;
                    //getchar();
                }
            }			
        }
    }
topResult.close();
    double score;
    int top = 1;
    //evaluation();

    
    std::fstream fout(dst+ "/log_time.txt", std::ios::app);

    double tv_ms = extract_feature_time*1.0/dir_size/1000;
    fout<<"extract_feature_time : "<<tv_ms<<"(ms)"<<std::endl;

    tv_ms = query_image_time_once*1.0/dir_size/1000;
    fout<<"query_image_time_once : "<<tv_ms<<"(ms)"<<std::endl;

    fout.close();

    for (int i=1; i <= rank; i++) {
        sum[i] /= dir_size;
    }
    cmc = sum;

    std::ofstream fresult(dst + "/result_writer.txt");

    //std::cout<<std::endl<< "soft evaluation result: "<<std::endl;
    fresult<<std::endl<< "soft evaluation result: "<<std::endl;
    for (int i =0; i <sum.size(); i++){         
        fresult<<i<<" : "<<sum[i]<<std::endl;       
        std::cout<<i<<":"<<sum[i]<<std::endl;        
    }
cout<<"map : "<<map / sum.size()<<endl;
    fresult.close();
}
