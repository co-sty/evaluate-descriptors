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
	Noise alter;
	alter.add(I,J);
	alter.msg();

	imshow("abc",J);

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

	// Testing
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
	Mat img_matches;
	drawMatches( I, keypointsI, J, keypointsJ, matches, img_matches );
	imshow("Matches", img_matches );

	waitKey(0);

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./evaluation ../data/a.jpg" << std::endl;
}
