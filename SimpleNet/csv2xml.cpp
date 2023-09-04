#include <opencv2\opencv.hpp>
#include <opencv2\opencv_modules.hpp> 
#include <opencv2\ml.hpp>
#include<iostream>


int main_csv2xml()
{
	auto mlData = cv::ml::TrainData::loadFromCSV("train.csv",0);//¶ÁÈ¡csvÎÄ¼þ
	cv::Mat data = cv::Mat(mlData->getTrainSamples(), true);
	std::cout << "Data have been read successfully!" << std::endl;
	//Mat double_data;
	//data.convertTo(double_data, CV_64F);
	
	cv::Mat input_ = data(cv::Rect(1, 1, 784, data.rows - 1)).t();
	cv::Mat label_ = data(cv::Rect(0, 1, 1, data.rows - 1));
	cv::Mat target_(10, input_.cols, CV_32F, cv::Scalar::all(0.));

	cv::Mat digit(28, 28, CV_32FC1);
	cv::Mat col_0 = input_.col(3);
	float label0 = label_.at<float>(3, 0);
	std::cout << label0;
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			digit.at<float>(i, j) = col_0.at<float>(i * 28 + j);
		}
	}

	for (int i = 0; i < label_.rows; ++i)
	{
		float label_num = label_.at<float>(i, 0);
		//target_.at<float>(label_num, i) = 1.;
		target_.at<float>((int)label_num, i) = label_num;
	}

	cv::Mat input_normalized(input_.size(), input_.type());
	for (int i = 0; i < input_.rows; ++i)
	{
		for (int j = 0; j < input_.cols; ++j)
		{
			//if (input_.at<double>(i, j) >= 1.)
			//{
			input_normalized.at<float>(i, j) = input_.at<float>(i, j) / 255.0f;
			//}
		}
	}

	std::string filename = "input_label_0-9.xml";
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "input" << input_normalized;
	fs << "target" << target_; // Write cv::Mat
	fs.release();


	cv::Mat input_1000 = input_normalized(cv::Rect(0, 0, 10000, input_normalized.rows));
	cv::Mat target_1000 = target_(cv::Rect(0, 0, 10000, target_.rows));

	std::string filename2 = "input_label_0-9_10000.xml";
	cv::FileStorage fs2(filename2, cv::FileStorage::WRITE);

	fs2 << "input" << input_1000;
	fs2 << "target" << target_1000; // Write cv::Mat
	fs2.release();

	return 0;
}
