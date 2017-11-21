
//#include "kernel.h"
#include "hog.hpp"

#include "assert.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/core/cuda.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;


class HOG
{
	static Ptr<HOG> create(Size win_size = Size(64, 128),
		Size block_size = Size(16, 16),
		Size block_stride = Size(8, 8),
		Size cell_size = Size(8, 8),
		int nbins = 9);
};


int main()
{
	//test::test_main();
	//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	cv::Mat img_rgb = imread("test.png");

	// Convert to C4
	cv::Mat img;
	cv::cvtColor(img_rgb, img, cv::COLOR_BGR2BGRA);

	cv::cuda::GpuMat d_img(img);

	// Convert train images into feature vectors (train table)
	cv::cuda::GpuMat descriptors, descriptors_by_cols;


	Ptr<HOG> hog;
	//hog = HOG::create();

	//hog = HOG::create();


	hog->setWinStride(Size(64, 128));

	//hog->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_ROW_BY_ROW);
	//hog->compute(d_img, descriptors);

	//hog->setDescriptorFormat(cv::cuda::HOG::DESCR_FORMAT_COL_BY_COL);
	//hog->compute(d_img, descriptors_by_cols);

	//// Check size of the result train table
	//wins_per_img_x = 3;
	//wins_per_img_y = 2;
	//blocks_per_win_x = 7;
	//blocks_per_win_y = 15;
	//block_hist_size = 36;
	//cv::Size descr_size_expected = cv::Size(blocks_per_win_x * blocks_per_win_y * block_hist_size,
	//	wins_per_img_x * wins_per_img_y);
	//ASSERT_EQ(descr_size_expected, descriptors.size());

	//// Check both formats of output descriptors are handled correctly
	//cv::Mat dr(descriptors);
	//cv::Mat dc(descriptors_by_cols);
	//for (int i = 0; i < wins_per_img_x * wins_per_img_y; ++i)
	//{
	//	const float* l = dr.rowRange(i, i + 1).ptr<float>();
	//	const float* r = dc.rowRange(i, i + 1).ptr<float>();
	//	for (int y = 0; y < blocks_per_win_y; ++y)
	//		for (int x = 0; x < blocks_per_win_x; ++x)
	//			for (int k = 0; k < block_hist_size; ++k)
	//				ASSERT_EQ(l[(y * blocks_per_win_x + x) * block_hist_size + k],
	//				r[(x * blocks_per_win_y + y) * block_hist_size + k]);
	//}
	system("pause");

}
