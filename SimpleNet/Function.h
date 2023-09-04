#pragma once
#include <opencv2\core\core.hpp>
#include <iostream>

namespace SimpleNet
{
	enum class func_type {
		sigmoid,
		tanh,
		relu
	};
	//sigmoid function
	cv::Mat sigmoid(const cv::Mat &x);

	//Tanh function
	cv::Mat tanh(const cv::Mat &x);

	//ReLU function
	cv::Mat ReLU(const cv::Mat &x);

	//Derivative function
	cv::Mat derivativeFunction(const cv::Mat& fx, func_type ft);

	//Objective function
	void calcLoss(const cv::Mat &output, const cv::Mat &target, cv::Mat &output_error, float &loss);

}
