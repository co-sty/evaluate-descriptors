#include "eval.hpp"

using namespace cv;
using namespace std;

Eval::Eval(vector<KeyPoint> k1_,
	         vector<KeyPoint> k2_,
	         vector< DMatch > matches_)
		 		 : k1(k1_), k2(k2_), matches(matches_)
{}


float Eval::putative_match_ratio()
{
	return ((float) matches.size()) * 2  / (float) (k1.size()+k2.size());
}

float Eval::precision()
{
	return ((float) correct_matches()) / (float) matches.size();
}

float Eval::matching_score()
{
	return ((float) correct_matches()) * 2  / ((float) (k1.size()+k2.size()) );
}

int Eval::correct_matches()
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

int Eval::correspondences()
{
	Point2f a,b;
	float count = 0;

	for(int i=0; i<k1.size(); i++)
	{
		a = k1[i].pt;
		for(int j=0; j<k2.size(); j++)
		{
			b = k2[j].pt;
			if(norm(Mat(a),Mat(b)) < eps)
				count++;
		}
	}

	return (int)count/2;
}
