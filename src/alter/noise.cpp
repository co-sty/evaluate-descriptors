#include "alter.hpp"

using namespace cv;
using namespace std;


//---------------------------------
//            Noise
//---------------------------------

Noise::Noise(const Mat& mat_ref_)
      : Alter(mat_ref_)
{
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
