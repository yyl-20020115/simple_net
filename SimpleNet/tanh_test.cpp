#include "Net.h"
#include <opencv2/opencv.hpp>

int main_tanh_test()
{
	//Get test samples and the label is 0--1
	cv::Mat test_input, test_label;
	int sample_number = 200;
	int start_position = 800;
	SimpleNet::GetInputLabel("data/input_label_1000.xml", test_input, test_label, sample_number, start_position);

	//convert label from 0---1 to -1---1,cause tanh function range is [-1,1]
	test_label = 2 * test_label - 1;

	//Load the trained net and test.
	SimpleNet::Net net;
	net.Load("models/model_tanh_800_200.xml");
	net.Test(test_input, test_label);

	//getchar();
	return 0;
}




