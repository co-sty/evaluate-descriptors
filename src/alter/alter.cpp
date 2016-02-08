#include "alter.hpp"

using namespace cv;
using namespace std;


//---------------------------------
//            Alter
//---------------------------------

Alter::Alter(const Mat& ref_img_)
      : ref_img(ref_img_)
{}

int Alter::add(const Mat& in, Mat& out)
{
  in.copyTo(out);
  return 0;
}

int Alter::rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2)
{
  rm_outliers(k2, k2);
  return 0;
}

int Alter::rm_outliers(vector<KeyPoint>& k1, vector<KeyPoint>& k2)
{
  k2 = k1;//.clone();

  remove_if(k1.begin(), k1.end(),
  [&](KeyPoint k){
    return (k.pt.x < Alter::ref_img.cols
      &&  k.pt.x >= 0
      &&  k.pt.y < Alter::ref_img.rows
      &&  k.pt.y >= 0);}
    );

  return 0;
}
/*
bool Alter::isInCanvas(KeyPoint k)
{
  return (k.pt.x < ref_img.cols
      &&  k.pt.x >= 0
      &&  k.pt.y < ref_img.rows
      &&  k.pt.y >= 0);
}
*/
int Alter::msg()
{
  cout << "No modification" <<endl;
  return 0;
}
