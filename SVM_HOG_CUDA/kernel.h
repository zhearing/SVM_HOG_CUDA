
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/cuda/common.hpp"

namespace test
{
	int test_main();
};

namespace HOG
{
	void set_up_constants(int nbins,
		int block_stride_x, int block_stride_y,
		int nblocks_win_x, int nblocks_win_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream);

	void compute_hists(int nbins,
		int block_stride_x, int block_stride_y,
		int height, int width,
		const cv::cuda::PtrStepSzf& grad, const cv::cuda::PtrStepSzb& qangle,
		float sigma,
		float* block_hists,
		int cell_size_x, int cell_size_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream);

	void normalize_hists(int nbins,
		int block_stride_x, int block_stride_y,
		int height, int width,
		float* block_hists,
		float threshold,
		int cell_size_x, int cell_size_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream);

	void classify_hists(int win_height, int win_width, int block_stride_y,
		int block_stride_x, int win_stride_y, int win_stride_x, int height,
		int width, float* block_hists, float* coefs, float free_coef,
		float threshold, int cell_size_x, int ncells_block_x, unsigned char* labels);

	void compute_confidence_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
		float* coefs, float free_coef, float threshold, int cell_size_x, int ncells_block_x, float *confidences);

	void extract_descrs_by_rows(int win_height, int win_width,
		int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x,
		int height, int width,
		float* block_hists,
		int cell_size_x, int ncells_block_x,
		cv::cuda::PtrStepSzf descriptors,
		const cudaStream_t& stream);
	void extract_descrs_by_cols(int win_height, int win_width,
		int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x,
		int height, int width,
		float* block_hists,
		int cell_size_x, int ncells_block_x,
		cv::cuda::PtrStepSzf descriptors,
		const cudaStream_t& stream);

	void compute_gradients_8UC1(int nbins,
		int height, int width, const cv::cuda::PtrStepSzb& img,
		float angle_scale,
		cv::cuda::PtrStepSzf grad, cv::cuda::PtrStepSzb qangle,
		bool correct_gamma,
		const cudaStream_t& stream);
	void compute_gradients_8UC4(int nbins,
		int height, int width, const cv::cuda::PtrStepSzb& img,
		float angle_scale,
		cv::cuda::PtrStepSzf grad, cv::cuda::PtrStepSzb qangle,
		bool correct_gamma,
		const cudaStream_t& stream);

	void resize_8UC1(const cv::cuda::PtrStepSzb& src, cv::cuda::PtrStepSzb dst);
	void resize_8UC4(const cv::cuda::PtrStepSzb& src, cv::cuda::PtrStepSzb dst);

};
