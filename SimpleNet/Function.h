#pragma once
#include <opencv2\core\core.hpp>
#include <iostream>

namespace SimpleNet
{
	enum class FuncType {
		Sigmoid,
		Tanh,
		ReLU
	};
	//sigmoid function
	cv::Mat Sigmoid(const cv::Mat &x);

	//Tanh function
	cv::Mat Tanh(const cv::Mat &x);

	//ReLU function
	cv::Mat ReLU(const cv::Mat &x);

	//Derivative function
	cv::Mat DerivativeFunction(const cv::Mat& fx, FuncType ft);

	//Objective function
	void CalcLoss(const cv::Mat &output, const cv::Mat &target, cv::Mat &output_error, float &loss);

}
