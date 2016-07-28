#include "connected_components.h"
#include<iostream>
#include<stdlib.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include"FsHelpers.h"

using namespace std;

/** DisjointSet **/
DisjointSet::DisjointSet() :
  m_disjoint_array(),
  m_subset_num(0)
{  }

DisjointSet::DisjointSet(int size) :
  m_disjoint_array(),
  m_subset_num(0)
{
  m_disjoint_array.reserve(size);
}

DisjointSet::~DisjointSet()
{  }

//add a new element, which is a subset by itself;
int DisjointSet::add()
{
  int cur_size = m_disjoint_array.size();
  m_disjoint_array.push_back(cur_size);
  m_subset_num ++;
  return cur_size;
}
//return the root of x
int DisjointSet::find(int x)
{
  if (m_disjoint_array[x] < 0 || m_disjoint_array[x] == x)
    return x;
  else {
    m_disjoint_array[x] = this->find(m_disjoint_array[x]);
    return m_disjoint_array[x];
  }
}
// point the x and y to smaller root of the two
void DisjointSet::unite(int x, int y)
{
  if (x==y) {
    return;
  }
  int xRoot = find(x);
  int yRoot = find(y);
  if (xRoot == yRoot)
    return;
  else if (xRoot < yRoot) {
    m_disjoint_array[yRoot] = xRoot;
  }
  else {
    m_disjoint_array[xRoot] = yRoot;
  }
  m_subset_num--;
}

int DisjointSet::getSubsetNum()
{
  return m_subset_num;
}

/** ConnectedComponent **/
ConnectedComponent::ConnectedComponent() :
  m_bb(0,0,0,0),
  m_pixel_count(0),
  m_pixels()
{
  m_pixels = std::make_shared< std::vector<cv::Point2i> > ();
}

ConnectedComponent::ConnectedComponent(int x, int y) :
  m_bb(x,y,1,1),
  m_pixel_count(1),
  m_pixels()
{
  m_pixels = std::make_shared< std::vector<cv::Point2i> > ();
}

ConnectedComponent::~ConnectedComponent(void)
{ }

void ConnectedComponent::addPixel(int x, int y) {
  m_pixel_count++;
  // new bounding box;
  if (m_pixel_count == 0) {
    m_bb = cv::Rect(x,y,1,1);
  }
  // extend bounding box if necessary
  else {
    if (x < m_bb.x ) {
      m_bb.width+=(m_bb.x-x);
      m_bb.x = x;
    }
    else if ( x > (m_bb.x+m_bb.width) ) {
      m_bb.width=(x-m_bb.x);
    }
    if (y < m_bb.y ) {
      m_bb.height+=(m_bb.y-y);
      m_bb.y = y;
    }
    else if ( y > (m_bb.y+m_bb.height) ) {
      m_bb.height=(y-m_bb.y);
    }
  }
  m_pixels->push_back(cv::Point(x,y));
}

int ConnectedComponent::getBoundingBoxArea(void) const {
  return (m_bb.width*m_bb.height);
}

cv::Rect ConnectedComponent::getBoundingBox(void) const {
  return m_bb;
}

std::shared_ptr< const std::vector<cv::Point2i> > ConnectedComponent::getPixels(void) const {
  return m_pixels;
}


int ConnectedComponent::getPixelCount(void) const {
  return m_pixel_count;
}

/** find connected components **/

void findCC2(const cv::Mat& src, std::vector<ConnectedComponent>& cc) {
  if ( src.empty()) {
      cout<<"empty"<<endl;
      return;
  }
  CV_Assert(src.type() == CV_8U);
  cc.clear();
  
  int total_pix = src.total();

  int* frame_label = new int [total_pix];
  if ( frame_label == NULL )
      cout<<"malloc fail "<<endl;

  DisjointSet labels(total_pix);

  int* root_map = new int [total_pix];
  if ( root_map == NULL )
      cout<<"MALLOC FAIL "<<endl;
  
  int x, y;
  const uchar* cur_p;
  const uchar* prev_p = src.ptr<uchar>(0);
  int left_val, up_val;
  int cur_idx, left_idx, up_idx;
  cur_idx = 0;
  
  //first logic loop
  for (y = 0; y < src.rows; y++ ) {
    cur_p = src.ptr<uchar>(y);
    for (x = 0; x < src.cols; x++, cur_idx++) {
      left_idx = cur_idx - 1;
      up_idx = cur_idx - src.size().width;
      if ( x == 0)
        left_val = 0;
      else
        left_val = cur_p[x-1];
      if (y == 0)
        up_val = 0;
      else
        up_val = prev_p[x];
      if (cur_p[x] > 0) {
        //current pixel is foreground and has no connected neighbors
        if (left_val == 0 && up_val == 0) {
          frame_label[cur_idx] = (int)labels.add();
          root_map[frame_label[cur_idx]] = -1;
        }
        //current pixel is foreground and has left neighbor connected
        else if (left_val != 0 && up_val == 0) {
          frame_label[cur_idx] = frame_label[left_idx];
        }
        //current pixel is foreground and has up neighbor connect
        else if (up_val != 0 && left_val == 0) {
          frame_label[cur_idx] = frame_label[up_idx];
        }
        //current pixel is foreground and is connected to left and up neighbors
        else {
          frame_label[cur_idx] = (frame_label[left_idx] > frame_label[up_idx]) ? frame_label[up_idx] : frame_label[left_idx];
          labels.unite(frame_label[left_idx], frame_label[up_idx]);
        }
      }//endif
      else {
        frame_label[cur_idx] = -1;
      }
    } //end for x
    prev_p = cur_p;
  }//end for y
  //second loop logic
  cur_idx = 0;
  int curLabel;
  int connCompIdx = 0;
  for (y = 0; y < src.size().height; y++ ) {
    for (x = 0; x < src.size().width; x++, cur_idx++) {
      curLabel = frame_label[cur_idx];
      if (curLabel != -1) {
        curLabel = labels.find(curLabel);
        if( root_map[curLabel] != -1 ) {
          cc[root_map[curLabel]].addPixel(x, y);
        }
        else {
          cc.push_back(ConnectedComponent(x,y));
          root_map[curLabel] = connCompIdx;
          connCompIdx++;
        }
      }
    }//end for x
  }//end for y
}


void findCC(const cv::Mat& src, std::vector<ConnectedComponent>& cc) {
if (src.empty()) return;
CV_Assert(src.type() == CV_8U);
cc.clear();
int total_pix = int(src.total());
int *frame_label = new int[total_pix];
DisjointSet labels(total_pix);
int *root_map = new int[total_pix];
int x, y;
const uchar* cur_p;
const uchar* prev_p = src.ptr<uchar>(0);
int left_val, up_val, up_left_val, up_right_val;
int cur_idx, left_idx, up_idx, up_left_idx, up_right_idx;
cur_idx = 0;
//first logic loop
for (y = 0; y < src.rows; y++) {
    cur_p = src.ptr<uchar>(y);
    for (x = 0; x < src.cols; x++, cur_idx++) {
        left_idx = cur_idx - 1;
        up_idx = cur_idx - src.size().width;
        up_left_idx = up_idx - 1;
        up_right_idx = up_idx + 1;

        if (x == 0)
        {
            left_val = 0;
        }
        else
        {
            left_val = cur_p[x - 1];
        }
        if (y == 0)
        {
            up_val = 0;
        }
        else
        {
            up_val = prev_p[x];
        }
        if (x == 0 || y == 0)
        {
            up_left_val = 0;
        }
        else
        {
            up_left_val = prev_p[x-1];
        }
        if (x == src.cols - 1 || y == 0)
        {
            up_right_val = 0;
        }
        else
        {
            up_right_val = prev_p[x+1];
        }

        if (cur_p[x] > 0) {
            //current pixel is foreground and has no connected neighbors
            if (left_val == 0 && up_val == 0 && up_left_val == 0 && up_right_val == 0) {
                frame_label[cur_idx] = (int)labels.add();
                root_map[frame_label[cur_idx]] = -1;
            }

            //Current pixel is foreground and has at least one neighbor
            else
            {
                vector<int> frame_lbl;
                frame_lbl.reserve(4);
                //Find minimal label
                int min_frame_lbl = INT_MAX;
                int valid_entries_num = 0;

                if (left_val != 0)
                {
                    frame_lbl.push_back(frame_label[left_idx]);
                    min_frame_lbl = min(min_frame_lbl, frame_label[left_idx]);
                    valid_entries_num++;
                }
                if (up_val != 0)
                {
                    frame_lbl.push_back(frame_label[up_idx]);
                    min_frame_lbl = min(min_frame_lbl, frame_label[up_idx]);
                    valid_entries_num++;
                }
                if (up_left_val != 0)
                {
                    frame_lbl.push_back(frame_label[up_left_idx]);
                    min_frame_lbl = min(min_frame_lbl, frame_label[up_left_idx]);
                    valid_entries_num++;
                }
                if (up_right_val != 0)
                {
                    frame_lbl.push_back(frame_label[up_right_idx]);
                    min_frame_lbl = min(min_frame_lbl, frame_label[up_right_idx]);
                    valid_entries_num++;
                }

                CV_Assert(valid_entries_num > 0);
                frame_label[cur_idx] = min_frame_lbl;

                //Unite if necessary
                if (valid_entries_num > 1)
                {
                    for (size_t i = 0; i < frame_lbl.size(); i++)
                    {
                        labels.unite(frame_lbl[i], min_frame_lbl);
                    }
                }
            }

        }//endif
        else {
            frame_label[cur_idx] = -1;
        }
    } //end for x
    prev_p = cur_p;
}//end for y
//second loop logic
cur_idx = 0;
int curLabel;
int connCompIdx = 0;
for (y = 0; y < src.size().height; y++) {
    for (x = 0; x < src.size().width; x++, cur_idx++) {
        curLabel = frame_label[cur_idx];
        if (curLabel != -1) {
            curLabel = labels.find(curLabel);
            if (root_map[curLabel] != -1) {
                cc[root_map[curLabel]].addPixel(x, y);
            }
            else {
                cc.push_back(ConnectedComponent(x, y));
                root_map[curLabel] = connCompIdx;
                connCompIdx++;
            }
        }
    }//end for x
}//end for y

//Free up allocated memory
delete[] frame_label;
delete[] root_map;
}

string atoiMy( int a  ) {
    stringstream ss;
    string b;
    ss<<a;
    ss>>b;
    return b;
}

int Process( const string& path, const string& savePath ) {
    if ( !FsHelpers::Exists( savePath ) )
        FsHelpers::MakeDirectory( savePath );
    cv::Mat img = cv::imread(path.c_str(),0 ) ;  
    if ( img.data == NULL )
        cout<<"image empty "<<endl;
    cv::Mat src;
    cv::threshold(img, src, 100, 255 , CV_THRESH_BINARY_INV) ;
    cv::imwrite("binary.jpg", src);
    
    vector<ConnectedComponent>cc;
    findCC( src, cc);

    for( int i = 1; i <= cc.size();i++ ){
        cv::Rect rect = cc[i].getBoundingBox();
        if ( rect.height < 10 || rect.width < 10 )
            continue;
        double ratio = double( rect.height ) / double( rect.width); 
        if ( ratio > 2.1 || ratio < 0.3 )
            continue;
        
        cout<<rect.x<<'\t'<<rect.y<<'\t'<<rect.width<<'\t'<<rect.height<<'\t'<<img.cols<<'\t'<<img.rows<<endl;
        cv::rectangle( img, cv::Point(rect.x,rect.y ), cv::Point( rect.x + rect.width, rect.y + rect.height ), cv::Scalar( 0 ,0,255), 2 );
        cv::Mat crop;
        try {
            crop = img( rect );
        } catch( cv::Exception ) {
            cv::rectangle( img, cv::Point(rect.x,rect.y ), cv::Point( rect.x + rect.width, rect.y + rect.height ), cv::Scalar( 0 ,0,255), 2 );
            cv::imwrite( "test.jpg", img );
            getchar();
        }
        string label = atoiMy(i);
        string Path = FsHelpers::CombinePath( savePath ,label);
        Path +=  ".jpg"; 
        cv::imwrite( Path.c_str(), crop );
    }
    cv::imwrite( "test.jpg", img );
    return 0 ; 
}

int ProcessC( const string& file, const string& savePath ) {
    if ( !FsHelpers::Exists( savePath ) )
        FsHelpers::MakeDirectory( savePath );
    vector<string> files;
    FsHelpers::GetFilesHasExtension( file , files, ".jpg",false);
    for ( size_t i = 0; i < files.size(); i++ ) {
        cout<<files[i]<<endl;
        std::string img_name = FsHelpers::GetFileNameWithoutExtension( files[i]);

        Process( files[i], savePath + img_name );
        //std::string img_name = FsHelpers::GetFileName(directories[i]);
    }
}
