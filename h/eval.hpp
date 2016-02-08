#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class Eval {
public:
	float eps = 3;	// tolerance for identical keypoints

	Eval(vector<KeyPoint> k1_, 		// keypoints from 1st img
	     vector<KeyPoint> k2_, 		// corrected keypoints, 2nd img
	     vector< DMatch > matches_);
	float putative_match_ratio();
	float precision();
	float matching_score();
	float recall();
	float entropy();

private:
	vector<KeyPoint> k1, k2;
	vector< DMatch > matches;
	int correct_matches();
	int correspondences();
};
