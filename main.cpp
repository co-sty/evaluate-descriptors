#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "eval.hpp"
#include "alter.hpp"
#include <stdlib.h>
#include <unordered_map>
#include <chrono>

using namespace cv;
using namespace std;

float noise_variance = 30.0;
int entropy_step = 50;
float entropy_var = 1.0;

unordered_map<string,Ptr<Feature2D> > methods({
	{"ORB",         ORB::create()},
	{"BRISK",       BRISK::create()},
	{"AKAZE",       AKAZE::create()},
	{"MSER",        MSER::create()}
});

string alterations[2] = {"noise","transf"};

string combinations[2][2] = {
		{"AKAZE","AKAZE"},
		{"AKAZE","ORB"}
	};

int alter_length = (int) sizeof(alterations)/sizeof(*alterations);
int comb_length = (int) sizeof(combinations)/sizeof(*combinations);

void readme();

Mat eval_method(
	const string& dir_name, 
	const string& alter_name, 
	const string& detector_name, 
	const string& descriptor_name,
	bool print
	);

int eval_all(const string &dir_name, const string &db_name);




/** @function main */
int main( int argc, char** argv )
{
	if(argc < 2)
	{
		readme();
		return -1;
	}

		// eval_method(argv[1],argv[4],argv[2],argv[3]);
	eval_all(argv[1],"data/res.xml");

return 0;
}


Mat eval_method(
	const string& dir_name, 
	const string& alter_name, 
	const string& detector_name, 
	const string& descriptor_name,
	bool print=true
	)
{
	// get filenames
	vector< String > filenames;
	glob(dir_name,filenames);

	Mat results = Mat::zeros(10,1,CV_32F);
	vector<string> result_keys;

	for(int i=0; i<filenames.size(); i++)
	{
		// load image
		Mat I = imread(filenames[i]);
		if(I.empty())
		{
			cout << filenames[i] << " ignored" << endl;
			continue;
		}
		else
		{
			if(print)
				cout << "evaluating " << filenames[i] << " ..." << endl;
		}

		// alter image
		Alter alter;
		if(strcmp(alter_name.c_str(),"noise"))
		{
			cout << "noise" <<endl;
			Noise alter(I.size(),noise_variance);	
		}
		else if(strcmp(alter_name.c_str(),"noise"))
		{
			cout << "Transf" <<endl;
			Transf alter(I.size());
		}
		else
		{
			Alter alter(I);
		}
		//Noise alter(I.size(),noise_variance);
		Mat J;
		alter.add(I,J);

		Ptr<Feature2D> detector = methods[detector_name];
		Ptr<Feature2D> descriptor = methods[descriptor_name];

		// detect
		// auto start = std::chrono::system_clock::now();
		vector<KeyPoint> keypointsI, keypointsJ;
		detector->detect(I, keypointsI);
		detector->detect(J, keypointsJ);
		if(print)
		{
			cout << "kpts : "
				<<keypointsI.size()
				<<" & "
				<<keypointsJ.size()
				<<endl;
		}

		// describe
		Mat descriptorsI, descriptorsJ;
		descriptor->compute( I, keypointsI, descriptorsI );
		descriptor->compute( J, keypointsJ, descriptorsJ );
		// auto end = std::chrono::system_clock::now();
		// auto elapsed = end - start;
		// cout << elapsed.count() << endl;

		// match
		BFMatcher matcher(NORM_L2);// EMD ?
		vector< DMatch > matches;
		matcher.match( descriptorsI, descriptorsJ, matches );

		// test
		vector<KeyPoint> keypointsJ_;
		alter.rm(keypointsJ, keypointsJ_);
		Eval eval(keypointsI, keypointsJ_, matches, I.size());
		result_keys = eval.result_keys;

		// accumulate
		results += eval.results(entropy_step,entropy_var);
	}

	results /= filenames.size();

	// Print results
	if(print)
	{
		cout << endl;
		cout << "============" << endl;
		cout << "Results : " << endl;
		cout << "detector : " << detector_name << endl;
		cout << "descriptor : " << descriptor_name << endl;
		for(int i=0; i<result_keys.size(); i++)
		{
			cout << result_keys[i] << endl;
			cout << results.at<float>(i) << endl;
		}
	}

	return results;
}


/*
	if(plot)
	{
		//----------------------------
		// Display Results
		string win_matches = "Matches",
		win_keypoints = "Keypoints (red=originals, green=rectified)";
		// matches
		namedWindow(win_matches,WINDOW_NORMAL);
		Mat img_matches, img_keypoints;
		drawMatches( I, keypointsI, J, keypointsJ, matches, img_matches, Scalar(100,100,100), Scalar(0,0,255) );
		imshow( win_matches, img_matches );
		// keypoints
		namedWindow(win_keypoints,WINDOW_NORMAL);
		drawKeypoints( I, keypointsI, img_keypoints, Scalar(0,0,255) );
		drawKeypoints( I, keypointsJ_, img_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_OVER_OUTIMG );
		imshow(win_keypoints, img_keypoints );

		waitKey(0);
	}
*/



int eval_all(const string &dir_name, const string &db_name)
{

	FileStorage db(db_name, FileStorage::WRITE);

int a = 1;

	db << "meta" << "{";
	db << "params" << a;
	db << "}";

	db << "data" << "{";

	for(int i=0; i<comb_length; i++)
	{
		for(int j=0; j<alter_length; j++)
			{
				db << "test" << "{";
				db << "detector" << combinations[i][0];
				db << "descriptor" << combinations[i][1];
				db << "alteration" << alterations[j];

				Mat results = eval_method(dir_name,
					alterations[j],
					combinations[i][0],
					combinations[i][1],
					true);
				
				db << "results" << results;
				cout << results;
				db << "}";
			}
	}		

	db << "}";

	return 0;
}


/** @function readme */
void readme()
{
	std::cout << " Usage: ./evaluation ../data/satellite/4.tiff <Detector_name> <Descriptor_name> <Alter_name>" << std::endl;
	std::cout << " where available descriptors/detectors are : AKAZE, BRISK, ORB" << endl;
	std::cout << " and available alterations are : noise and affine" << endl;
}

// ./evaluation data/img AKAZE AKAZE noise

/*
AKAZE AKAZE
AKAZE ORB
AKAZE BRISK
ORB ORB
ORB BRISK
BRISK BRISK
BRISK ORB
*/
