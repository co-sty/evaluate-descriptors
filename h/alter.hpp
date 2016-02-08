#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// Parent Class

class Alter {
public:
  Alter(const Mat& mat_ref_);
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  int rm_outliers(vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  RNG rng;
private:
  const Mat& ref_img;
};

// Children

class Noise : public Alter {
public:
  Noise(const Mat& mat_ref_);
  virtual int add(const Mat& in, Mat& out);
  virtual int msg();
  float mean, var;
};

class Transf : public Alter {
public:
  Transf(const Mat& mat_ref_);
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  Mat warp_mat;
};
