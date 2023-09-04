#include "Net.h"
#include <opencv2/opencv.hpp>


int main_test()
{
	//Set neuron number of every layer
	std::vector<int> layer_neuron_num = { 784,100,10 };

	// Initialise Net and weights
	SimpleNet::Net net;
	net.InitNet(layer_neuron_num);
	net.InitWeights(0, 0., 0.01);
	net.InitBias(cv::Scalar(0.05));

	//Get test samples and test samples 
	cv::Mat input, label,test_input,test_label;
	int sample_number = 200;
	SimpleNet::GetInputLabel("data/input_label_1000.xml",input, label, sample_number);
	SimpleNet::GetInputLabel("data/input_label_1000.xml", test_input, test_label, 200, 800);

	//Set loss threshold,learning rate and activation function
	float loss_threshold = 0.5;
	net.learning_rate = 0.3f;
	net.output_interval = 2;
	net.activation_function = SimpleNet::FuncType::Sigmoid;

	//convert label from 0---1 to -1---1,cause tanh function range is [-1,1]
	//label = 2 * label - 1;

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.Train(input, label, loss_threshold,true);
	net.Test(test_input, test_label);

	//Save the model
	net.Save("models/model_sigmoid_800_200_test.xml");

//	getchar();
	return 0;

}




