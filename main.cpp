#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "eval.hpp"
#include "alter.hpp"
#include <stdlib.h>
#include <unordered_map>
#include <chrono>

using namespace cv;
using namespace std;
using namespace xfeatures2d;

// Globals

float noise_variance = 30.0;
int entropy_step = 50;
float entropy_var = 1.0;

unordered_map<string,Ptr<Feature2D> > methods({
	{"SIFT",        SIFT::create()},
	{"SURF",        SURF::create()},
	{"ORB",         ORB::create()},
	// {"BRIEF",       BRIEF::create()},
	{"BRISK",       BRISK::create()},
	{"AKAZE",       AKAZE::create()},
	// {"FREAK",       FREAK::create()},
	{"MSER",        MSER::create()}
});

string alterations[2] = {"noise","transf"};

string combinations[4][2] = {
		{"BRISK","BRISK"},
		{"ORB","ORB"},
		{"AKAZE","SURF"},
		{"AKAZE","AKAZE"}
	};

vector <string> result_keys = {
		"putative match ratio",
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

int alter_length = (int) sizeof(alterations)/sizeof(*alterations);
int comb_length = (int) sizeof(combinations)/sizeof(*combinations);

int t_detection, 
	t_description, 
	t_matching;


// Function declarations

void readme();

Mat eval_method(
	const string& dir_name, 
	const string& alter_name, 
	const string& detector_name, 
	const string& descriptor_name,
	bool print
	);

int eval_all(const string &dir_name, const string &db_name);

Alter* alteration_picker(Size size, String name)
{
	Alter* alter = NULL;
	if(!strcmp(name.c_str(),"noise"))
	{
		alter = new Noise( size , noise_variance );	
	}
	else if(!strcmp(name.c_str(),"transf"))
	{
		alter = new Transf( size );
	}
	else
	{
		cerr << "no transform";
		alter = new Alter(size);
	}	

	return alter;
}

// 


/** @function main */
int main( int argc, char** argv )
{
	if(argc == 1)
	{
		// print usage
		readme();
		return -1;
	}
	else if(argc == 5 && !strcmp(argv[4],"show"))
	{
		// display the keypoints for the given methods
		// show_method(argv[1],argv[2],argv[3]);

	}
	else if(argc == 3 && !strcmp(argv[2],"all"))
	{
		// evaluate all the methods
		cout << "evaluating all";
		eval_all(argv[1],"data/res.xml");
	}
	else if(argc == 5)
	{
		// evaluate the given methods, with the given alteration
		eval_method(argv[1],argv[4],argv[2],argv[3],true);
	}

	return 0;
}


/** @function eval_method */
Mat eval_method(
	const string& dir_name, 
	const string& alter_name, 
	const string& detector_name, 
	const string& descriptor_name,
	bool print = true
	)
{
	// get filenames
	vector< String > filenames;
	glob(dir_name,filenames);

	Mat results = Mat::zeros(10,1,CV_32F);
	vector<string> result_keys;
	int count = 0;
	t_detection = 0;
	t_description = 0;
	t_matching = 0;

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
			count++;
		}

		// alter image
		Alter* alter = alteration_picker(I.size(),alter_name);
		Mat J;
		alter -> add(I,J);

		imshow("a",I);
		imshow("b",J);
		waitKey(1);

		Ptr<Feature2D> detector = methods[detector_name];
		Ptr<Feature2D> descriptor = methods[descriptor_name];

		// detect
		auto start = std::chrono::system_clock::now();
			vector<KeyPoint> keypointsI, keypointsJ;
			detector -> detect(I, keypointsI);
			detector -> detect(J, keypointsJ);
		auto end = std::chrono::system_clock::now();
		auto elapsed = chrono::duration_cast<std::chrono::milliseconds> (end - start);
		t_detection += elapsed.count();
		if(print)
		{
			cout << "kpts : "
				<<keypointsI.size()
				<<" & "
				<<keypointsJ.size()
				<<endl;
		}

		// describe
		start = std::chrono::system_clock::now();
			Mat descriptorsI, descriptorsJ;
			descriptor->compute( I, keypointsI, descriptorsI );
			descriptor->compute( J, keypointsJ, descriptorsJ );
		end = std::chrono::system_clock::now();
		elapsed = chrono::duration_cast<std::chrono::milliseconds> (end - start);
		t_description += elapsed.count();

		// match
		start = std::chrono::system_clock::now();
			BFMatcher matcher(NORM_L2);// EMD ?
			vector< DMatch > matches;
			matcher.match( descriptorsI, descriptorsJ, matches );
		end = std::chrono::system_clock::now();
		elapsed = chrono::duration_cast<std::chrono::milliseconds> (end - start);
		t_matching += elapsed.count();

		// test
		vector<KeyPoint> keypointsJ_;
		alter -> rm(keypointsJ, keypointsJ_);
		Eval eval(keypointsI, keypointsJ_, matches, I.size());
		result_keys = eval.result_keys;

		// accumulate
		results += eval.results(entropy_step,entropy_var);//results_temp;

		delete alter;
	}

	results 		/= count;
	t_detection 	/= count;
	t_description 	/= count;
	t_matching 		/= count;

	// Print results
	if(print)
	{
		cout << endl;
		cout << "============" << endl;
		cout << "Results : " << endl;
		cout << "detector : " << detector_name << endl;
		cout << "descriptor : " << descriptor_name << endl;
		cout << "time : " << t_detection<<"\t"<<t_description<<"\t"<<t_matching 
			 << "ms/(detect+desc.+matching)" << endl;
		for(int i=0; i<result_keys.size(); i++)
		{
			cout << result_keys[i] << endl;
			cout << results.at<float>(i) << endl;
		}
	}

	return results;
}


/** @function eval_all */
int eval_all(const string &dir_name, const string &db_name)
{

	FileStorage db(db_name, FileStorage::WRITE);

	int a = 1;

	db << "meta" << "{";
	db << "result_keys" << result_keys;
	db << "}";

	db << "data" << "{";

	for(int i=0; i<comb_length; i++)
	{
		// for(int j=0; j<alter_length; j++)
			// {
				db << "test" << "{";
					db << "detector" << combinations[i][0];
					db << "descriptor" << combinations[i][1];
					db << "alteration" << "noise";//alterations[j];

					Mat results = eval_method(dir_name,
												"noise",//alterations[j],
												combinations[i][0],
												combinations[i][1],
												true
											);


					cout << endl << t_detection + t_description + t_matching << endl;
					
					db << "results" << results;

					vector <int> times = {t_detection, t_description, t_matching};
					db << "times" << times;

				db << "}";
			// }
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
