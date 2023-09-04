#include"Function.h"

//sigmoid function
cv::Mat SimpleNet::Sigmoid(const cv::Mat& x)
{
	cv::Mat exp_x, fx;
	cv::exp(-x, exp_x);
	fx = 1.0 / (1.0 + exp_x);
	return fx;
}

//tanh function
cv::Mat SimpleNet::Tanh(const cv::Mat& x)
{
	cv::Mat exp_x_, exp_x, fx;
	cv::exp(-x, exp_x_);
	cv::exp(x, exp_x);
	fx = (exp_x - exp_x_) / (exp_x + exp_x_);
	return fx;
}

//ReLU function
cv::Mat SimpleNet::ReLU(const cv::Mat& x)
{
	cv::Mat fx = x;
	for (int i = 0; i < fx.rows; i++)
	{
		for (int j = 0; j < fx.cols; j++)
		{
			if (fx.at<float>(i, j) < 0)
			{
				fx.at<float>(i, j) = 0;
			}
		}
	}
	return fx;
}

//Derivative function
cv::Mat SimpleNet::DerivativeFunction(const cv::Mat& fx, FuncType ft)
{
	cv::Mat dx;
	if (ft == FuncType::Sigmoid)
	{
		dx = SimpleNet::Sigmoid(fx).mul((1 - Sigmoid(fx)));
	}
	else if (ft == FuncType::Tanh)
	{
		cv::Mat tanh_2;
		pow(Tanh(fx), 2., tanh_2);
		dx = 1 - tanh_2;
	}
	else if (ft == FuncType::ReLU)
	{
		dx = fx;
		for (int i = 0; i < fx.rows; i++)
		{
			for (int j = 0; j < fx.cols; j++)
			{
				if (fx.at<float>(i, j) > 0)
				{
					dx.at<float>(i, j) = 1;
				}
			}
		}
	}
	return dx;
}

//Objective function
void SimpleNet::CalcLoss(const cv::Mat& output, const  cv::Mat& target, cv::Mat& output_error, float& loss)
{
	if (target.empty())
	{
		std::cout << "Can't find the target cv::Matrix" << std::endl;
		return;
	}
	output_error = target - output;
	cv::Mat err_sqrare;
	pow(output_error, 2., err_sqrare);
	cv::Scalar err_sqr_sum = sum(err_sqrare);
	loss =(float)( err_sqr_sum[0] / (float)(output.rows));
}


