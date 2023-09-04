#include "Net.h"
#include <opencv2/opencv.hpp>

int main_sigmoid_test()
{
	//Get test samples and the label is 0--1
	cv::Mat test_input, test_label;
	int sample_number = 200;
	int start_position = 800;
	SimpleNet::GetInputLabel("data/input_label_1000.xml", test_input, test_label, sample_number, start_position);

	//Load the trained net and test.
	SimpleNet::Net net;
	net.Load("models/model_sigmoid_800_200.xml");
	net.Test(test_input, test_label);

	//getchar();
	return 0;
}




