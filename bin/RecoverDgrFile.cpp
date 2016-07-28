/*************************************************************************
	> File Name: RecoverDgrFile.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Fri 01 Jul 2016 03:18:46 PM CST
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<string.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"FsHelpers.h"
#include<sstream>
#include<cstring>

using namespace std;

#define MAX_ILLUSTR_LEN 128 

const string filePath[2] ={ "gallery","probe" };

struct DGR_HEADER { 
    int iHdSize; // size of header: 54+strlen(illustr)+1 (there is a '\0' at the end of illustr) 
    char szFormatCode[8]; // "DGR" 
    char szIllustr[MAX_ILLUSTR_LEN]; // text of arbitrary length. "#......\0" 
    char szCodeType[20]; // "ASCII", "GB", "SJIS" etc 
    short sCodeLen; // 1, 2, 4, etc 
    short sBitApp; // "1 or 8 bit per pixel" etc 
}; 
        
//the annotation information of a word 
struct WORD_INFO { 
    unsigned char *pWordLabel; // the pointer to the word label (GB code) 
    short sTop; // the top coordinate of a word image 
    short sLeft; // the left coordinate of a word image 
    short sHei; // the height of a word image 
    short sWid; // the width of a word image 
}; 
              
//the annotation information of a text line 
struct LINE_INFO { 
    int iWordNum; // the word number in a text line 
    WORD_INFO *pWordInfo; // the pointer to the annotation information of the words in a text line 
};  

// the annotation information of document image 
struct DOC_IMG { 
    int iImgHei; // the height of the document image 
    int iImgWid; // the width of the document image 
    int iLineNum; // the text line number in the document image 
    LINE_INFO *pLineInfo; // the pointer to the annotation information of the text lines 
    unsigned char *pDocImg; // the pointer to image data buffer 
}; 
                      
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
// // 
// read annotation information from *.dgr file // 
// recovery the * dgr file to document image data // 
// // 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
void ReaddgrFile2Img(FILE *fp, const string& path, DOC_IMG& docImg, const string& phase ) // fp is the file pointer to *.dgr file 
{
    string phasePath ;
    if ( !strcmp( phase.c_str(), "train" ) ) {
        phasePath = filePath[0];
    } else
        phasePath = filePath[1];
    string rootPath = "../dataset/";
    string pathDir = rootPath + phasePath ;
    string savePath = pathDir + "/" + path ;
    if ( !FsHelpers::Exists( savePath )) {
        FsHelpers::MakeDirectories( savePath ); 
    }
        
    DGR_HEADER dgrHead; 

    // read the head information of the *.dgr file 
    fread(&dgrHead.iHdSize, 4, 1, fp);

    fread(dgrHead.szFormatCode, 8, 1, fp); 
    fread(dgrHead.szIllustr, (dgrHead.iHdSize - 36), 1, fp); 
    fread(dgrHead.szCodeType, 20, 1, fp); 
    fread(&dgrHead.sCodeLen, 2, 1, fp); 
    fread(&dgrHead.sBitApp, 2, 1, fp); 

    // read the height and width of the document image 
    fread(&docImg.iImgHei, 4, 1, fp); 
    fread(&docImg.iImgWid, 4, 1, fp); 

    // allocate memory for the document image data 
    docImg.pDocImg = new unsigned char [docImg.iImgHei * docImg.iImgWid]; 
    memset(docImg.pDocImg, 0xff, docImg.iImgHei * docImg.iImgWid);  
    // allocate memory for the annotation information of text lines 
    fread(&docImg.iLineNum, 4, 1, fp); 
    docImg.pLineInfo = new LINE_INFO [docImg.iLineNum]; 

    int i, j, m, n; 
    unsigned char *pTmpData; 
    int iTmpDataSize; 
    short iTmpDataTop; 
    short iTmpDataLeft; 
    short iTmpDataHei; 
    short iTmpDataWid; 
            
    // recovery the document image line by line 
    for(i = 0; i < docImg.iLineNum; i++) { 
        // read the word number in the i-th text line 
        fread(&docImg.pLineInfo[i].iWordNum, 4, 1, fp); 
        docImg.pLineInfo[i].pWordInfo = new WORD_INFO [ docImg.pLineInfo[i].iWordNum ];

        //// read the annotation information of every word in the i-th text line 
        for(j = 0; j < docImg.pLineInfo[i].iWordNum; j++) { 
            docImg.pLineInfo[i].pWordInfo[j].pWordLabel = new unsigned char [dgrHead.sCodeLen]; 
            
            if ( docImg.pLineInfo[i].pWordInfo[j].pWordLabel  == NULL )
                cout<<"alloc fail";
            
            fread(docImg.pLineInfo[i].pWordInfo[j].pWordLabel, dgrHead.sCodeLen, 1, fp); 
            fread(&docImg.pLineInfo[i].pWordInfo[j].sTop, 2, 1, fp); 
            fread(&docImg.pLineInfo[i].pWordInfo[j].sLeft, 2, 1, fp); 
            fread(&docImg.pLineInfo[i].pWordInfo[j].sHei, 2, 1, fp); 
            fread(&docImg.pLineInfo[i].pWordInfo[j].sWid, 2, 1, fp); 

            iTmpDataTop = docImg.pLineInfo[i].pWordInfo[j].sTop; 
            iTmpDataLeft = docImg.pLineInfo[i].pWordInfo[j].sLeft; 
            iTmpDataHei = docImg.pLineInfo[i].pWordInfo[j].sHei; 
            iTmpDataWid = docImg.pLineInfo[i].pWordInfo[j].sWid; 
            pTmpData = new unsigned char [iTmpDataHei * iTmpDataWid]; 
            fread(pTmpData, iTmpDataHei * iTmpDataWid, 1, fp); 
            // write the the word data image to the document image data 
            
            for(m = 0; m < iTmpDataHei; m++) { 
                for(n = 0; n < iTmpDataWid; n++) { 
                    if(pTmpData[m * iTmpDataWid + n] != 255) { 
                        docImg.pDocImg[(m + iTmpDataTop) * docImg.iImgWid + n + iTmpDataLeft]  = pTmpData[m * iTmpDataWid + n]; 
                    } 
                } 
            } 
        
            cv::Mat image = cv::Mat( iTmpDataHei, iTmpDataWid, CV_8UC1, pTmpData );
            string line,word;
            stringstream istr;
            istr<<i;
            istr>>line;
            istr.clear();
            istr<<j;
            istr>>word;
            string name = savePath + "/" + line + "_" + word + ".jpg";
            //cout<<name<<endl;
            cv::imwrite(name , image );
            delete [] pTmpData;
            delete [] docImg.pLineInfo[i].pWordInfo[j].pWordLabel;
        } 
        delete [] docImg.pLineInfo[i].pWordInfo;
    }
    fclose( fp );

string fileName = string( "../dataset/Test_Dgr/") + path;
cout<<fileName<<endl;
fp = fopen( fileName.c_str(), "rb+");
if ( fp == NULL )
    cout<<"OPEN FAIL"<<endl;
fread(&dgrHead.iHdSize, 4, 1, fp);

    fread(dgrHead.szFormatCode, 8, 1, fp); 
    fread(dgrHead.szIllustr, (dgrHead.iHdSize - 36), 1, fp); 
    fread(dgrHead.szCodeType, 20, 1, fp); 
    fread(&dgrHead.sCodeLen, 2, 1, fp); 
    fread(&dgrHead.sBitApp, 2, 1, fp); 

    // read the height and width of the document image 
    fread(&docImg.iImgHei, 4, 1, fp); 
    fread(&docImg.iImgWid, 4, 1, fp); 
    fread(&docImg.iLineNum, 4, 1, fp); 
cout<<docImg.iImgHei<<endl;
cout<<docImg.iImgWid<<endl;
cout<<docImg.iLineNum<<endl;

cv::Mat img = cv::Mat( docImg.iImgHei, docImg.iImgWid, CV_8UC1, docImg.pDocImg );
for(i = 0; i < docImg.iLineNum; i++) { 
    // read the word number in the i-th text line 
    int iWordNum;
    fread(&iWordNum, 4, 1, fp); 
    cout<< iWordNum<<endl;
    try{
        docImg.pLineInfo[i].pWordInfo = new WORD_INFO [ docImg.pLineInfo[i].iWordNum ];
    } catch( bad_alloc) {
        cout<<iWordNum<<endl;
        getchar();
    }
    //// read the annotation information of every word in the i-th text line 
    for(j = 0; j < iWordNum; j++) { 
        unsigned char* pWordLabel = new unsigned char [dgrHead.sCodeLen]; 
        if ( pWordLabel  == NULL )
            cout<<"alloc fail";
        
        fread( pWordLabel, dgrHead.sCodeLen, 1, fp); 
        
        short Top,Left,Hei,Wid;
        fread(&Top, 2, 1, fp); 
        fread(&Left, 2, 1, fp); 
        fread(&Hei, 2, 1, fp); 
        fread(&Wid, 2, 1, fp); 
            unsigned char* temp = new unsigned char [ Hei * Wid]; 
            fread( temp , Hei * Wid, 1, fp); 

        //cout<<Top<<'\t'<<Left<<'\t'<<Top + Hei <<'\t'<<Left + Wid<<endl;
        cv::rectangle( img, cv::Point( Left,Top ), cv::Point( Left + Wid, Top + Hei ), cv::Scalar(0,0,255) ,2); 
    }
}
cv::imwrite("test.jpg",img );
getchar();
} 

int RestoreImage ( const string& path, DOC_IMG& docImg, const string& phase ) {
    string phasePath ;
    string rootPath = "../dataset/";
    if ( !strcmp( phase.c_str(), "train" ) ) {
        phasePath = filePath[0];
    } else
        phasePath = filePath[1];

    string pathDir = rootPath + phasePath + "_whole/";
    if ( !FsHelpers::Exists( pathDir ) ) {
        FsHelpers::MakeDirectories( pathDir ); 
    }
    
    int wight = docImg.iImgWid;
    int height = docImg.iImgHei;
    unsigned char* img = docImg.pDocImg;
    //cout<<"height "<<height<<'\t'<<"width "<<wight<<endl;
    cv::Mat image = cv::Mat(height, wight, CV_8UC1, img);
    
    string savePath = pathDir + path + ".jpg" ;
    //cout<<savePath<<endl;
    cv::imwrite( savePath.c_str(), image );
    return 0;
}

int main( int argv,char** argc ) {
    if ( argv != 3 )
        cout<<"input error "<<endl;

    string dirs = argc[1];
    string phase = argc[2];
    string id;

    vector<string> files;
    FsHelpers::GetFilesHasExtension(dirs, files, ".dgr",false);

    for ( size_t i = 0; i < files.size(); i++ ) {
        string fileName = files[i];
        std::string name = FsHelpers::GetFileName(files[i]);
        
        int length = name.size();
        string pageIdx = name.substr( length - 6,2 );
        if ( !strcmp ( phase.c_str(), "train") ) 
            id = "16";
        else
            id = "17";

        if ( strcmp ( pageIdx.c_str(), id.c_str() ) ) {
            continue;
        }
        
        cout<<fileName<<endl;

        FILE* fp = fopen( fileName.c_str(), "rb+");
        if ( fp == NULL )
            cout<<"file "<<fileName<<" open error"<<endl;
        
        DOC_IMG docImg; 
        ReaddgrFile2Img( fp , name, docImg ,phase ) ;// fp is the file pointer to *.dgr file 
   
        RestoreImage( name , docImg, phase );
    }
    return 0;
}

