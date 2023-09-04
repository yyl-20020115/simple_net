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
		FuncType activation_function = FuncType::Sigmoid;
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
		void InitNet(const std::vector<int>& layer_neuron_num_);

		//Initialise the weights matrices.
		void InitWeights(int type = 0, double a = 0.0, double b = 0.1);

		//Initialise the bias matrices.
		void InitBias(const cv::Scalar& bias);

		//Forward
		void Forward();

		//Forward
		void Backward();

		//Train,use accuracy_threshold
		void Train(const cv::Mat& input, const cv::Mat& target, float accuracy_threshold);

		//Train,use loss_threshold
		void Train(const cv::Mat& input, const cv::Mat& target_, float loss_threshold, bool draw_loss_curve = false);

		//Test
		void Test(const cv::Mat &input, const cv::Mat &target_);

		//Predict,just one sample
		int Predict(const cv::Mat &input);

		//Predict,more  than one samples
		std::vector<int> Predicts(const cv::Mat &input);

		//Save model;
		void Save(const std::string& filename);

		//Load model;
		void Load(const std::string& filename);

	protected:
		//initialise the weight matrix.if type =0,Gaussian.else uniform.
		void InitWeight(const cv::Mat &dst, int type, double a, double b);

		//Activation function
		cv::Mat ActivationFunction(const cv::Mat &x, FuncType ft);

		//Compute delta error
		void ComputeDeltaError();

		//Update weights
		void UpdateWeights();
	};

	//Get sample_number samples in XML file,from the start column. 
	void GetInputLabel(const std::string& filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);

	// Draw loss curve
	void DrawCurve(cv::Mat& board, const std::vector<double>& points);
}
