#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

class Alter {
public:
  Alter();
  virtual int add(const Mat& in, Mat& out);
  virtual int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  //virtual int msg();
};

class Noise : public Alter {
public:
  Noise();
  virtual int add(const Mat& in, Mat& out);
  //int rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2);
  virtual int msg();
  float mean, var;
};
