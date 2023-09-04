#include "Net.h"
#include <opencv2/opencv.hpp>

int main_relu_test()
{
	//Get test samples and the label is 0--9
	cv::Mat test_input, test_label;
	int sample_number = 200;
	int start_position = 800;
	SimpleNet::GetInputLabel("data/input_label_0-9_1000.xml", test_input, test_label, sample_number, start_position);

	//Load the trained net and test.
	SimpleNet::Net net;
	net.Load("models/model_ReLU_800_200.xml");
	net.Test(test_input, test_label);

	//getchar();
	return 0;
}




