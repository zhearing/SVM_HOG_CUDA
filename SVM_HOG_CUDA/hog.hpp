
#include "kernel.h"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core.hpp"
#include "driver_types.h"

using namespace cv;

namespace hog
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

class CV_EXPORTS HOG : public Algorithm
{
public:
	enum
	{
		DESCR_FORMAT_ROW_BY_ROW,
		DESCR_FORMAT_COL_BY_COL
	};

	/** @brief Creates the HOG descriptor and detector.

	@param win_size Detection window size. Align to block size and block stride.
	@param block_size Block size in pixels. Align to cell size. Only (16,16) is supported for now.
	@param block_stride Block stride. It must be a multiple of cell size.
	@param cell_size Cell size. Only (8, 8) is supported for now.
	@param nbins Number of bins. Only 9 bins per cell are supported for now.
	*/
	static Ptr<HOG> create(Size win_size = Size(64, 128),
		Size block_size = Size(16, 16),
		Size block_stride = Size(8, 8),
		Size cell_size = Size(8, 8),
		int nbins = 9);

	//! Gaussian smoothing window parameter.
	virtual void setWinSigma(double win_sigma) = 0;
	virtual double getWinSigma() const = 0;

	//! L2-Hys normalization method shrinkage.
	virtual void setL2HysThreshold(double threshold_L2hys) = 0;
	virtual double getL2HysThreshold() const = 0;

	//! Flag to specify whether the gamma correction preprocessing is required or not.
	virtual void setGammaCorrection(bool gamma_correction) = 0;
	virtual bool getGammaCorrection() const = 0;

	//! Maximum number of detection window increases.
	virtual void setNumLevels(int nlevels) = 0;
	virtual int getNumLevels() const = 0;

	//! Threshold for the distance between features and SVM classifying plane.
	//! Usually it is 0 and should be specfied in the detector coefficients (as the last free
	//! coefficient). But if the free coefficient is omitted (which is allowed), you can specify it
	//! manually here.
	virtual void setHitThreshold(double hit_threshold) = 0;
	virtual double getHitThreshold() const = 0;

	//! Window stride. It must be a multiple of block stride.
	virtual void setWinStride(Size win_stride) = 0;
	virtual Size getWinStride() const = 0;

	//! Coefficient of the detection window increase.
	virtual void setScaleFactor(double scale0) = 0;
	virtual double getScaleFactor() const = 0;

	//! Coefficient to regulate the similarity threshold. When detected, some
	//! objects can be covered by many rectangles. 0 means not to perform grouping.
	//! See groupRectangles.
	virtual void setGroupThreshold(int group_threshold) = 0;
	virtual int getGroupThreshold() const = 0;

	//! Descriptor storage format:
	//!   - **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.
	//!   - **DESCR_FORMAT_COL_BY_COL** - Column-major order.
	virtual void setDescriptorFormat(int descr_format) = 0;
	virtual int getDescriptorFormat() const = 0;

	/** @brief Returns the number of coefficients required for the classification.
	*/
	virtual size_t getDescriptorSize() const = 0;

	/** @brief Returns the block histogram size.
	*/
	virtual size_t getBlockHistogramSize() const = 0;

	/** @brief Sets coefficients for the linear SVM classifier.
	*/
	virtual void setSVMDetector(InputArray detector) = 0;

	/** @brief Returns coefficients of the classifier trained for people detection.
	*/
	virtual Mat getDefaultPeopleDetector() const = 0;

	/** @brief Performs object detection without a multi-scale window.

	@param img Source image. CV_8UC1 and CV_8UC4 types are supported for now.
	@param found_locations Left-top corner points of detected objects boundaries.
	@param confidences Optional output array for confidences.
	*/
	virtual void detect(InputArray img,
		std::vector<Point>& found_locations,
		std::vector<double>* confidences = NULL) = 0;

	/** @brief Performs object detection with a multi-scale window.

	@param img Source image. See cuda::HOGDescriptor::detect for type limitations.
	@param found_locations Detected objects boundaries.
	@param confidences Optional output array for confidences.
	*/
	virtual void detectMultiScale(InputArray img,
		std::vector<Rect>& found_locations,
		std::vector<double>* confidences = NULL) = 0;

	/** @brief Returns block descriptors computed for the whole image.

	@param img Source image. See cuda::HOGDescriptor::detect for type limitations.
	@param descriptors 2D array of descriptors.
	@param stream CUDA stream.
	*/
	//virtual void compute(InputArray img,
	//	OutputArray descriptors,
	//	Stream& stream = Stream::Null()) = 0;
};


