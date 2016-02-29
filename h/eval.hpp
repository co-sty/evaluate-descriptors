#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class Eval {
public:
	float eps = 2;	// tolerance for identical keypoints

	Eval(const vector<KeyPoint> &k1_, 		// keypoints from 1st img
	     const vector<KeyPoint> &k2_, 		// corrected keypoints, 2nd img
	     const vector< DMatch > &matches_,
	     const Size size);
	// measures
	float putative_match_ratio();
	float precision();
	float matching_score();
	float recall();
	float entropy(float step, float sigma);
	//
	int nb_correspondences();
	int nb_correct_matches();
	// output
	Mat results(float step, float sigma);
	int print_results();
	vector <string> result_keys = {"putative match ratio",
		"precision",
		"matching_score",
		"recall",
		"entropy",
		"number of keypoints 1",
		"number of keypoints 2",
		"number of correct matches",
		"number of correspondences",
		"tolerance"
	};

private:
	vector<KeyPoint> k1, k2;
	vector< DMatch > matches;
	int n_keypoints;
	Size ref_size;
	int step_x, step_y;
	
};
