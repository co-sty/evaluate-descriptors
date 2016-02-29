#include "alter.hpp"

using namespace cv;
using namespace std;


//---------------------------------
//            Noise
//---------------------------------

Noise::Noise(Size ref_size_)
: Alter(ref_size_), mean(0)
{
	var = rng.uniform(1,10);
}

Noise::Noise(Size ref_size_, float var_)
: Alter(ref_size_), mean(0), var(var_)
{}

int Noise::add(const Mat& in, Mat& out)
{
	// generate noise
	Mat noise_img = in.clone();
	randn(noise_img,  mean, var);

	// apply noise
	out = in + noise_img;

	return 0;
}

int Noise::rm(const vector<KeyPoint>& k1, vector<KeyPoint>& k2)
{
	k2 = k1;
	return 0;
}

int Noise::msg()
{
	cout << "adding N(" << mean << "," << var << ") noise" << endl;
	return 0;
}
