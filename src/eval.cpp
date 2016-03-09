#include "eval.hpp"

using namespace cv;
using namespace std;

Eval::Eval(const vector<KeyPoint> &k1_,
		const vector<KeyPoint> &k2_,
		const vector< DMatch > &matches_,
		const Size ref_size_)
: k1(k1_), k2(k2_), matches(matches_), ref_size(ref_size_)
{
	n_keypoints = k1.size();
//	n_keypoints = max(k1.size(),k2.size());
//	n_keypoints = min(k1.size(),k2.size());
//	n_keypoints = (int) ((float)(k1.size() + k2.size()))/2;
}

float Eval::putative_match_ratio()
{
	return ((float) matches.size()) / (float) n_keypoints;
}

float Eval::precision()
{
	return ((float) nb_correct_matches()) / (float) matches.size();
}

float Eval::matching_score()
{
	return ((float) nb_correct_matches()) / (float) n_keypoints;
}

float Eval::recall()
{
	return ((float) nb_correct_matches()) / (float) nb_correspondences();
}

int Eval::nb_correct_matches()
{
	Point2f a,b;
	int count = 0;

	for(int i=0; i<matches.size(); i++)
	{
		a = k1.at(matches[i].queryIdx).pt;
		b = k2.at(matches[i].trainIdx).pt;
		if(norm(Mat(a),Mat(b)) < eps)
			count++;
	}

	return count;
}

int Eval::nb_correspondences()
{
	Point2f a,b;
	float count = 0;

	vector<KeyPoint> ka,kb;
	// select ka, kb
	if(k1.size()>k2.size())
	{
		ka = k1;
		kb = k2;
	}
	else
	{
		ka = k2;
		kb = k1;
	}
/*
	ka = k1;
	kb = k2;
*/
	int j_min, min_dist;

	for(int i=0; i<ka.size(); i++)
	{
		a = ka[i].pt;
		min_dist = 100;
		for(int j=0; j<kb.size(); j++)
		{
			if(j!=i)
			{
				b = kb[j].pt;
				if(norm(Mat(a),Mat(b)) < min_dist)
				{
					min_dist = norm(Mat(a),Mat(b));
				}
			}
		}
		if(min_dist<eps)
			count++;

	}

	return (int)count;
}

float Eval::entropy(float step, float sigma)
{
	Mat B = Mat::zeros(ref_size.width/step,ref_size.height/step,CV_32F);
	Mat logB, BlogB;
	float* pointerB;

	for(int px=step/2; px<ref_size.width; px+=step)
	{
		pointerB = B.ptr<float>(px);
		for(int py=step/2; py<ref_size.height; py+=step)
		{
			for(int i=0; i<k1.size(); i++)
			{
				float d = norm( Point2f(px,py) - k1[i].pt );
				pointerB[py] += (float) exp(-d/(2*sigma));
			}
//			cout<<pointerB[py]<<endl ;
		}
//		cout<<B<<endl;
	}

	log(B,logB);
	multiply(B,logB,BlogB);
	float entropy = 0.0;
	Scalar S = -sum(BlogB);
//	cout<<B<<endl;

	return entropy;
}

Mat Eval::results(float step, float sigma)
{
	Mat results(10,1,CV_32F);
	float* ptr_results = results.ptr<float>(0);

	ptr_results[0] = putative_match_ratio();
	ptr_results[1] = precision();
	ptr_results[2] = matching_score();
	ptr_results[3] = recall();
	ptr_results[4] = 0.0;//entropy(step,sigma);
	ptr_results[5] = (float) k1.size();
	ptr_results[6] = (float) k2.size();
	ptr_results[7] = (float) matches.size();
	ptr_results[8] = (float) nb_correct_matches();
	ptr_results[9] = (float) nb_correspondences();
	ptr_results[10] = (float) eps;

	return results;
}

int Eval::print_results()
{
	cout << endl
		<< "Results"
		<< endl
		<< "----------------------------------"
		<< endl
		<< "[ " << k1.size() << " and " << k2.size() << " keypoints, "
		<< matches.size() << " matches ]"
		<< endl
		<< "putative match ratio :\t"
		<< putative_match_ratio()
		<< endl
		<< "precision :\t\t"
		<< precision() << " (" << nb_correct_matches() 
		<< " correct matches % " << eps << " px)"
		<< endl
		<< "matching score :\t"
		<< matching_score()
		<< endl
		<< "recall :\t\t"
		<< recall() << " (" << nb_correspondences() 
		<< " corresp. % " << eps << " px)"
		<< endl
		<< "entropy :\t\t"
//		<< entropy(50.0,1000.0)
		<< endl
		<< "----------------------------------";

	return 0;
}
