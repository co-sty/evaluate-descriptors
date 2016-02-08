#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "eval.hpp"
#include "alter.hpp"

using namespace cv;
using namespace std;

// ./evaluation ../data/a.jpg ../data/results.yml

void readme();

/** @function main */
int main( int argc, char** argv )
{
	Mat I = imread(argv[1]);//imread("../data/pietons/train/+_cropped/1.jpg");
	Mat J;
	Transf alter(I);
	alter.add(I,J);
	alter.msg();

	// Detect
	Ptr<Feature2D> detector = ORB::create();
	vector<KeyPoint> keypointsI, keypointsJ;
	detector->detect(I, keypointsI);
	detector->detect(J, keypointsJ);

	// Describe
	Ptr<Feature2D> descriptor = ORB::create();
	Mat descriptorsI, descriptorsJ;
	descriptor->compute( I, keypointsI, descriptorsI );
	descriptor->compute( J, keypointsJ, descriptorsJ );

	// Match
	BFMatcher matcher(NORM_L2);
	vector< DMatch > matches;
	matcher.match( descriptorsI, descriptorsJ, matches );

	// Test
	vector<KeyPoint> keypointsJ_;
	alter.rm(keypointsJ, keypointsJ_);
	Eval eval(keypointsI, keypointsJ_, matches);
	cout << endl
			 << "Results"
		   << endl
			 << "----------------------------------"
			 << endl
		 	 << "putative match ratio :\t"
			 << eval.putative_match_ratio()
			 << endl
		   << "precision :\t\t"
			 << eval.precision()
			 << endl
	     << "matching score :\t"
			 << eval.matching_score()
			 << endl
			 << "(tolerance of " << eval.eps << " px)"
			 << endl;

	// Display Results
  string win_matches = "Matches",
         win_keypoints1 = "Keypoints1",
         win_keypoints2 = "Keypoints2";
	Mat img_matches, img_keypoints1, img_keypoints2;
	drawMatches( I, keypointsI, J, keypointsJ, matches, img_matches, Scalar(100,100,100), Scalar(0,0,255) );
	imshow(win_matches, img_matches );
	drawKeypoints( I, keypointsI, img_keypoints1, Scalar(0,0,255) );
	drawKeypoints( I, keypointsJ_, img_keypoints2, Scalar(0,255,0) );
	imshow(win_keypoints1, img_keypoints1 );
	imshow(win_keypoints2, img_keypoints2 );

	waitKey(0);

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./evaluation ../data/a.jpg" << std::endl;
}
