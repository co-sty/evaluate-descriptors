#include "alter.hpp"

using namespace cv;
using namespace std;


//---------------------------------
//            Alter
//---------------------------------

Alter::Alter() 
      : ref_size(Mat::zeros(1,1,CV_32F).size())
{}

Alter::Alter(Size ref_size_)
      : ref_size(ref_size_)
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
  k2 = k1;//.copy();

  remove_if(k1.begin(), k1.end(),
  [&](KeyPoint k){
    return (k.pt.x < ref_size.width
      &&  k.pt.x >= 0
      &&  k.pt.y < ref_size.height
      &&  k.pt.y >= 0);}
    );

  return 0;
}

int Alter::msg()
{
  cout << "No modification" <<endl;
  return 0;
}
