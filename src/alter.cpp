#include "alter.hpp"

using namespace cv;
using namespace std;


Alter::Alter(){}

int Alter::add(const Mat& in, Mat& out)
{
  in.copyTo(out);
  return 0;
}

int Alter::rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2)
{
  k2 = k1;
  return 0;
}

Noise::Noise()
{
  RNG rng(12345);
  mean = 0;
  var = rng.uniform(-10,10);
}
int Noise::add(const Mat& in, Mat& out)
{
  // generate noise
  Mat noise_img = in.clone();
  randn(noise_img,  mean, var);

  // apply noise
  out = in + noise_img;

  return 0;
}

int Noise::msg()
{
  cout << "adding N(" << mean << "," << var << ") noise" << endl;
  return 0;
}
