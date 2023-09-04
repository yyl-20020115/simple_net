#include "Net.h"
#include <opencv2/opencv.hpp>

int main_relu_train()
{
	//Set neuron number of every layer
	std::vector<int> layer_neuron_num = { 784,100,100,10 };

	// Initialise Net and weights
	SimpleNet::Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);

	//Get test samples and test samples 
	cv::Mat input, label, test_input, test_label;
	int sample_number = 800;

	//SimpleNet::get_input_label("data/input_label_1-10_1000.xml", input, label, sample_number);
	//SimpleNet::get_input_label("data/input_label_1-10_1000.xml", test_input, test_label, 200, 800);

	SimpleNet::get_input_label("data/input_label_0-9_1000.xml", input, label, sample_number);
	SimpleNet::get_input_label("data/input_label_0-9_1000.xml", test_input, test_label, 200, 800);

	//Set loss threshold,learning rate and activation function
	float loss_threshold = 412;
	net.learning_rate = 0.002f;
	net.output_interval = 2;
	net.activation_function = "ReLU";

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.train(input, label, loss_threshold, true);
	net.test(test_input, test_label);

	//Save the model
	net.save("models/model_ReLU_800_200.xml");

	//getchar();
	return 0;

}




