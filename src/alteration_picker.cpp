#include "alteration_picker.hpp"

Alter* alteration_picker(String name)
{
	Alter* alter = NULL;
	if(!strcmp(name.c_str(),"noise"))
	{
		cout << "noise" << endl;
		alter = new Noise( I.size() , noise_variance );	
	}
	else if(!strcmp(name.c_str(),"noise"))
	{
		cout << "transform" << endl;
		alter = new Transf( I.size() );
	}
	else
	{
		cout << "none";
		alter = new Alter(I.size());
	}	

	return alter;
}