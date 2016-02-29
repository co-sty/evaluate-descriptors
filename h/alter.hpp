#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// Parent Class

class Alter {
public:
  Alter();
  Alter(Size ref_size);
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  int rm_outliers(vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  RNG rng;
private:
  Size ref_size;
};

// Children

class Noise : public Alter {
public:
  Noise(Size ref_size);
  Noise(Size ref_size, float var_);
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  float mean, var;
};

class Transf : public Alter {
public:
  Transf(Size ref_size);
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  Mat warp_mat;
};
