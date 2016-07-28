/*************************************************************************
	> File Name: main.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Thu 30 Jun 2016 04:57:28 PM CST
 ************************************************************************/

#include<iostream>
#include"writer.h"
#include<sstream>

using namespace std;

int ProcessC( const string& file, const string& savePath ) ;

int main( int argv, char** argc ) {
    const char* model = "models_writer_CASIA";
    const char* galler_dir = "CASIA_gallery";
    const char* test_data = "CASIA_probe";
    
    //TEST_CASIA_My( model , galler_dir , test_data);
    ProcessC("./test", "./test_cut/");
    return 0;
}
