#include "alter.hpp"

using namespace cv;
using namespace std;

//---------------------------------
//        Transformation
//---------------------------------


Transf::Transf(Size ref_size_)
      : Alter(ref_size_)
{
  int c = ref_size_.width,
      r = ref_size_.height;
  Point2f srcTri[3],
          dstTri[3];
  float top_pc = 0.9,
        bot_pc = 0.1;

  srcTri[0] = Point2f( 0,0 );
  srcTri[1] = Point2f( c - 1.f, 0 );
  srcTri[2] = Point2f( 0, r - 1.f );

  dstTri[0] = Point2f( 0.1f,0 );
  dstTri[1] = Point2f( c-0.99f, 0.01f );
  dstTri[2] = Point2f( 0, r-1.f );
/*
  dstTri[0] = Point2f( rng.uniform(0.f,c*0.1f), rng.uniform(0.f,r*bot_pc) );
  dstTri[1] = Point2f( rng.uniform((c-1.f)*top_pc,c-1.f), rng.uniform(0.f,(r-1.f)*top_pc) );
  dstTri[2] = Point2f( rng.uniform(0.f,c*bot_pc), rng.uniform((r-1.f)*top_pc,r-1.f) );
*/
  warp_mat = getAffineTransform( srcTri, dstTri );
}

int Transf::add(const Mat& in, Mat& out)
{

  Mat warp_dst = Mat::zeros( in.rows, in.cols, in.type() );
  warpAffine( in, warp_dst, warp_mat, warp_dst.size() );

  out = warp_dst.clone();

  return 0;
}

int Transf::rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2)
{
  vector<KeyPoint> k(k1);
  vector<Point2f> p1;

  Mat i_warp_mat;
  invertAffineTransform(warp_mat,i_warp_mat);

  for (int i=0; i<k1.size(); i++)
    p1.push_back(k1[i].pt);

  transform(p1, p1, i_warp_mat);

  for (int i=0; i<k.size(); i++)
    k[i].pt = p1[i];

  k2 = k;
  rm_outliers(k2,k2);
  return 0;
}

int Transf::msg()
{
  cout << "warping with matrix :"
       << endl
       << warp_mat
       << endl;
  return 0;
}
