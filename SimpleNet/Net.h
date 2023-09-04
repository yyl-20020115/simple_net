#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Function.h"

namespace SimpleNet
{
	class Net
	{
	public:
        //Integer vector specifying the number of neurons in each layer including the input and output layers.
		std::vector<double> loss_vec;
		std::vector<int> layer_neuron_num;
		func_type activation_function = func_type::sigmoid;
		int output_interval = 10;
		float learning_rate = 0.0f; 
		float accuracy = 0.0f;
		float fine_tune_factor = 1.01f;

	protected:
		std::vector<cv::Mat> layer;
		std::vector<cv::Mat> weights;
		std::vector<cv::Mat> bias;
		std::vector<cv::Mat> delta_err;

		cv::Mat output_error;
		cv::Mat target;
		cv::Mat board;
		float loss = 0.0f;

	public:
		Net();
		~Net();
	public:
		//Initialize net:genetate weights matrices¡¢layer matrices and bias matrices
		// bias default all zero
		void initNet(const std::vector<int>& layer_neuron_num_);

		//Initialise the weights matrices.
		void initWeights(int type = 0, double a = 0.0, double b = 0.1);

		//Initialise the bias matrices.
		void initBias(const cv::Scalar& bias);

		//Forward
		void forward();

		//Forward
		void backward();

		//Train,use accuracy_threshold
		void train(const cv::Mat& input, const cv::Mat& target, float accuracy_threshold);

		//Train,use loss_threshold
		void train(const cv::Mat& input, const cv::Mat& target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void test(const cv::Mat &input, const cv::Mat &target_);

		//Predict,just one sample
		int predict_one(const cv::Mat &input);

		//Predict,more  than one samples
		std::vector<int> predict(const cv::Mat &input);

		//Save model;
		void save(const std::string& filename);

		//Load model;
		void load(const std::string& filename);

	protected:
		//initialise the weight matrix.if type =0,Gaussian.else uniform.
		void initWeight(const cv::Mat &dst, int type, double a, double b);

		//Activation function
		cv::Mat activationFunction(const cv::Mat &x, func_type ft);

		//Compute delta error
		void deltaError();

		//Update weights
		void updateWeights();
	};

	//Get sample_number samples in XML file,from the start column. 
	void get_input_label(const std::string& filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);

	// Draw loss curve
	void draw_curve(cv::Mat& board, const std::vector<double>& points);
}