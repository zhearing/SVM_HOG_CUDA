
#include "kernel.h"

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/warp_shuffle.hpp"

#include <stdio.h>

namespace test
{
	cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

	__global__ void addKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}

	int test_main()
	{
		const int arraySize = 5;
		const int a[arraySize] = { 1, 2, 3, 4, 5 };
		const int b[arraySize] = { 10, 20, 30, 40, 50 };
		int c[arraySize] = { 0 };

		// Add vectors in parallel.
		cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4]);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}

	// Helper function for using CUDA to add vectors in parallel.
	cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
	{
		int *dev_a = 0;
		int *dev_b = 0;
		int *dev_c = 0;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, size >> >(dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	Error:
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		return cudaStatus;
	}
}


namespace HOG
{
	/*常量内存
	// 位置：设备内存
	// 形式：关键字__constant__添加到变量声明中。如__constant__ float s[10]; 。
	// 目的：为了提升性能。常量内存采取了不同于标准全局内存的处理方式。在某些情况下，用常量内存替换全局内存能有效地减少内存带宽。
	// 特点：常量内存用于保存在核函数执行期间不会发生变化的数据。变量的访问限制为只读。NVIDIA硬件提供了64KB的常量内存。不再需要cudaMalloc()或者cudaFree(), 而是在编译时，静态地分配空间。
	// 要求：当我们需要拷贝数据到常量内存中应该使用cudaMemcpyToSymbol()，而cudaMemcpy()会复制到全局内存。
	// 性能提升的原因：
	对常量内存的单次读操作可以广播到其他的“邻近”线程。这将节约15次读取操作。（为什么是15，因为“邻近”指半个线程束，一个线程束包含32个线程的集合。）
	常量内存的数据将缓存起来，因此对相同地址的连续读操作将不会产生额外的内存通信量。
	*/

	// using CUDA to hog
	__constant__ int cnbins;				// 直方图bin的数量(投票箱的个数)
	__constant__ int cblock_stride_x;		// x方向块的滑动步长，大小只支持是单元格cell_size大小的倍数
	__constant__ int cblock_stride_y;		//
	__constant__ int cnblocks_win_x;		// x方向 每个window中的block数
	__constant__ int cnblocks_win_y;		// 
	__constant__ int cncells_block_x;		// x方向 每个block中的cell数
	__constant__ int cncells_block_y;		//
	__constant__ int cblock_hist_size;		// 每个block的直方图大小
	__constant__ int cblock_hist_size_2up;	// 典型的gpu线程数
	__constant__ int cdescr_size;			// HOG特征向量的维数
	__constant__ int cdescr_width;			//特征向量的width


	/* 返回最接近的两个上限，仅适用于
	典型的GPU线程数（pert block）值 */
	int power_2up(unsigned int n)
	{
		if (n <= 1) return 1;
		else if (n <= 2) return 2;
		else if (n <= 4) return 4;
		else if (n <= 8) return 8;
		else if (n <= 16) return 16;
		else if (n <= 32) return 32;
		else if (n <= 64) return 64;
		else if (n <= 128) return 128;
		else if (n <= 256) return 256;
		else if (n <= 512) return 512;
		else if (n <= 1024) return 1024;
		return -1; // Input is too big
	}

	/* 返回nblocks的最大值 */
	int max_nblocks(int nthreads, int ncells_block = 1)
	{
		int threads = nthreads * ncells_block;
		if (threads * 4 <= 256)
			return 4;
		else if (threads * 3 <= 256)
			return 3;
		else if (threads * 2 <= 256)
			return 2;
		else
			return 1;
	}

	/*
	// nbins：直方图bin的数量，目前每个单元格Cell只支持9个
	// block_stride_x：width方向block的滑动步长，大小只支持单元格cell_size大小的倍数
	//
	// nblocks_win_x：blocks_per_win.width
	//
	// ncells_block_x：cells_per_block_.width
	//
	*/
	void set_up_constants(int nbins,
		int block_stride_x, int block_stride_y,
		int nblocks_win_x, int nblocks_win_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream)
	{
		cudaSafeCall(cudaMemcpyToSymbolAsync(cnbins, &nbins, sizeof(nbins), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_stride_x, &block_stride_x, sizeof(block_stride_x), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_stride_y, &block_stride_y, sizeof(block_stride_y), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cnblocks_win_x, &nblocks_win_x, sizeof(nblocks_win_x), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cnblocks_win_y, &nblocks_win_y, sizeof(nblocks_win_y), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cncells_block_x, &ncells_block_x, sizeof(ncells_block_x), 0, cudaMemcpyHostToDevice, stream));
		cudaSafeCall(cudaMemcpyToSymbolAsync(cncells_block_y, &ncells_block_y, sizeof(ncells_block_y), 0, cudaMemcpyHostToDevice, stream));

		int block_hist_size = nbins * ncells_block_x * ncells_block_y;
		cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_hist_size, &block_hist_size, sizeof(block_hist_size), 0, cudaMemcpyHostToDevice, stream));

		//最接近的上限，给gpu per block
		int block_hist_size_2up = power_2up(block_hist_size);
		cudaSafeCall(cudaMemcpyToSymbolAsync(cblock_hist_size_2up, &block_hist_size_2up, sizeof(block_hist_size_2up), 0, cudaMemcpyHostToDevice, stream));

		int descr_width = nblocks_win_x * block_hist_size;
		cudaSafeCall(cudaMemcpyToSymbolAsync(cdescr_width, &descr_width, sizeof(descr_width), 0, cudaMemcpyHostToDevice, stream));

		int descr_size = descr_width * nblocks_win_y;
		cudaSafeCall(cudaMemcpyToSymbolAsync(cdescr_size, &descr_size, sizeof(descr_size), 0, cudaMemcpyHostToDevice, stream));
	}


	/*
	法1.accumulate using atomics
		即，将最后一句变成
		atomicAdd(&(d_bins[myBin]), 1);
		但是对于atomics的方法而言，不管GPU多好，并行线程数都被限制到histogram个数N，也就是最多只有N个线程并行。 
	法2. local memory + reduce 
		设置n个并行线程，每个线程都有自己的local histogram（一个长为bin数的vector）
		即每个local histogram都被一个thread顺序访问，所以这样没有shared memory，即便没有用atomics也不会出现read-modify-write问题。
		然后，我们将这n个histogram进行合并（即加和），可以通过reduce实现。 
	法3. sort then reduce by key 
		将数据组织成key-value对，key为histogram bin，value为1，
		将其按key排序，然后对相同key进行reduce求和，就可以得到histogram中的每个bin的总数。
		ics的方法而言，不管GPU多好，并行线程数都被限制到histogram个数N，也就是最多只有N个线程并行。
	*/

	//----------------------------------------------------------------------------
	// 直方图计算
	//
	// CUDA内核来计算直方图
	template <int nblocks> // 单个GPU线程块处理的直方图块的数量
	__global__ void compute_hists_kernel_many_blocks(const int img_block_width, const cv::cuda::PtrStepf grad,
		const cv::cuda::PtrStepb qangle, float scale, float* block_hists,
		int cell_size, int patch_size, int block_patch_size,
		int threads_cell, int threads_block, int half_cell_size)
	{
		const int block_x = threadIdx.z; // 一共4个block
		const int cell_x = threadIdx.x / threads_cell;  // cell_x索引 0~1
		const int cell_y = threadIdx.y; // cell_y索引 0~1
		const int cell_thread_x = threadIdx.x & (threads_cell - 1);//   0~32 & 1111

		// blockIdx.y是gpu块在Y方向上的索引，threadIdx.x是线程在X方向上的索引，threadIdx.y是线程在Y方向上的索引
		// blockDim.x是gpu块在X方向上的线程数，blockDim.y是gpu块在Y方向上的线程数，数据在GPU里是按照行优先排列
		if (blockIdx.x * blockDim.z + block_x >= img_block_width) // block计算完成
			return;

		extern __shared__ float smem[];
		float* hists = smem;
		float* final_hist = smem + cnbins * block_patch_size * nblocks; // +9*48*4

		// patch_size means that patch_size pixels affect on block's cell 如何理解？
		// 一行有12个pixel影响cell
		if (cell_thread_x < patch_size) //12
		{
			// 滑动窗口的坐标
			const int offset_x = (blockIdx.x * blockDim.z + block_x) * cblock_stride_x +
				half_cell_size * cell_x + cell_thread_x; // imageblock的x索引 * 8 + 4 * cell的x索引
			const int offset_y = blockIdx.y * cblock_stride_y + half_cell_size * cell_y; // y索引 * 8 + 4 * cell的y索引

			const float* grad_ptr = grad.ptr(offset_y) + offset_x * 2; // 输出梯度
			const unsigned char* qangle_ptr = qangle.ptr(offset_y) + offset_x * 2; // 输入弧度


			float* hist = hists + patch_size * (cell_y * blockDim.z * cncells_block_y +
				cell_x + block_x * cncells_block_x) +
				cell_thread_x; // + 12 * (cell的y索引 * 1 * 2 + cell的x索引 + 第几个block * 2) + cell_thread_x
			for (int bin_id = 0; bin_id < cnbins; ++bin_id)
				hist[bin_id * block_patch_size * nblocks] = 0.f; //初始化hist[] 9*48*4

			//(dist_x, dist_y) : distance between current pixel in patch and cell's center
			const int dist_x = -half_cell_size + (int)cell_thread_x - half_cell_size * cell_x; // -4 + cell_thread_x - 4 * cell_x索引

			const int dist_y_begin = -half_cell_size - half_cell_size * (int)threadIdx.y;
			for (int dist_y = dist_y_begin; dist_y < dist_y_begin + patch_size; ++dist_y)
			{
				float2 vote = *(const float2*)grad_ptr;
				uchar2 bin = *(const uchar2*)qangle_ptr;

				grad_ptr += grad.step / sizeof(float);
				qangle_ptr += qangle.step;

				//(dist_center_x, dist_center_y) : distance between current pixel in patch and block's center
				int dist_center_y = dist_y - half_cell_size * (1 - 2 * cell_y);
				int dist_center_x = dist_x - half_cell_size * (1 - 2 * cell_x);

				//二维高斯分布，好抽象啊，看不懂
				float gaussian = ::expf(-(dist_center_y * dist_center_y +
					dist_center_x * dist_center_x) * scale);

				float interp_weight = ((float)cell_size - ::fabs(dist_y + 0.5f)) *
					((float)cell_size - ::fabs(dist_x + 0.5f)) / (float)threads_block;

				hist[bin.x * block_patch_size * nblocks] += gaussian * interp_weight * vote.x;
				hist[bin.y * block_patch_size * nblocks] += gaussian * interp_weight * vote.y;
			}

			// 归约直方图？？？
			volatile float* hist_ = hist;
			for (int bin_id = 0; bin_id < cnbins; ++bin_id, hist_ += block_patch_size * nblocks)
			{
				if (cell_thread_x < patch_size / 2) hist_[0] += hist_[patch_size / 2];
				if (cell_thread_x < patch_size / 4 && (!((patch_size / 4) < 3 && cell_thread_x == 0)))
					hist_[0] += hist_[patch_size / 4];
				if (cell_thread_x == 0)
					final_hist[((cell_x + block_x * cncells_block_x) * cncells_block_y + cell_y) * cnbins + bin_id]
					= hist_[0] + hist_[1] + hist_[2];
			}
		}

		__syncthreads();

		float* block_hist = block_hists + (blockIdx.y * img_block_width +
			blockIdx.x * blockDim.z + block_x) *
			cblock_hist_size;

		//从final_hist复制到block_hist
		int tid;
		if (threads_cell < cnbins)
		{
			tid = (cell_y * cncells_block_y + cell_x) * cnbins + cell_thread_x;
		}
		else
		{
			tid = (cell_y * cncells_block_y + cell_x) * threads_cell + cell_thread_x;
		}
		if (tid < cblock_hist_size)
		{
			block_hist[tid] = final_hist[block_x * cblock_hist_size + tid];
			if (threads_cell < cnbins && cell_thread_x == (threads_cell - 1))
			{
				for (int i = 1; i <= (cnbins - threads_cell); ++i)
				{
					block_hist[tid + i] = final_hist[block_x * cblock_hist_size + tid + i];
				}
			}
		}
	}


	/*
	// nbins：直方图bin的数量，目前每个单元格Cell只支持9个
	// block_stride_x：x方向块的滑动步长，大小只支持是单元格cell_size大小的倍数

	// 源图像只支持CV_8UC1和CV_8UC4数据类型
	// height：输入图像行数rows
	// width：输入图像列数cols

	// grad：输出梯度（两通道），记录每个像素所属bin对应的权重的矩阵，为幅值乘以权值。这个权值是关键，也很复杂：包括高斯权重，三次插值的权重，在本函数中先只考虑幅值和相邻bin间的插值权重
	// qangle：输入弧度（两通道），记录每个像素角度所属的bin序号的矩阵,均为2通道,为了线性插值
	// sigma：winSigma，高斯滤波窗口的参数
	// *block_hists：block_hists.ptr<float>
	*/

	// 声明变量，并用计算的blocks数调用kernel
	void compute_hists(int nbins,
		int block_stride_x, int block_stride_y,
		int height, int width,
		const cv::cuda::PtrStepSzf& grad, const cv::cuda::PtrStepSzb& qangle,
		float sigma,
		float* block_hists,
		int cell_size_x, int cell_size_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream)
	{
		const int ncells_block = ncells_block_x * ncells_block_y; // 4
		const int patch_side = cell_size_x / 4; // 2
		const int patch_size = cell_size_x + (patch_side * 2); // 8+4=12
		// 不共享的block数，共享的反复计算，可以存到shared memory
		const int block_patch_size = ncells_block * patch_size; // 4*12=48
		const int threads_cell = power_2up(patch_size); // 16
		const int threads_block = ncells_block * threads_cell; // 4*16=64
		const int half_cell_size = cell_size_x / 2; // 8/2=4

		// x方向block个数，相邻block之间会有重叠，y方向同理
		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) /
			block_stride_x; // 15
		int img_block_height = (height - ncells_block_y * cell_size_y + block_stride_y) /
			block_stride_y; // 7

		/*
		// fuction:divUp(int total, int grain)
		// return:(total + grain - 1) / grain;
		*/
		const int nblocks = max_nblocks(threads_cell, ncells_block); // 4
		dim3 grid(cv::cuda::device::divUp(img_block_width, nblocks), img_block_height); // 4,7
		dim3 threads(threads_cell * ncells_block_x, ncells_block_y, nblocks); // 16 * 2, 2 ,4

		// 预计算高斯空间Window参数
		float scale = 1.f / (2.f * sigma * sigma);

		int hists_size = (nbins * ncells_block * patch_size * nblocks) * sizeof(float); // 9*4*12*4=1728  *4字节
		int final_hists_size = (nbins * ncells_block * nblocks) * sizeof(float); // 9*4*4=144 *4字节
		int smem = hists_size + final_hists_size; // 1872 *4字节

		/*
		// 核函数只能在主机端调用，调用时必须申明执行参数
		// <<<>>>运算符内是核函数的执行参数，告诉编译器运行时如何启动核函数，用于说明内核函数中的线程数量，以及线程是如何组织的

		// 参数grid用于定义整个grid的维度和尺寸，即一个grid有多少个block，为dim3类型
		// Dim3 grid(grid.x, grid.y, 1)表示grid中每行有grid.x个block，每列有grid.y个block，第三维恒为1(目前一个核函数只有一个grid)
		// 整个grid中共有grid.x*grid.y个block，其中grid.x和grid.y最大值为65535

		// 参数threads用于定义一个block的维度和尺寸，即一个block有多少个thread，为dim3类型
		// Dim3 threads(threads.x, threads.y, threads.z)表示整个block中每行有threads.x个thread，每列有threads.y个thread，高度为threads.z。threads.x和threads.y最大值为1024，threads.z最大值为62
		// 一个block中共有threads.x*threads.y*threads.z个thread

		// 参数smem是一个可选参数，用于设置每个block除了静态分配的shared Memory以外，最多能动态分配的shared memory大小，单位为byte。不需要动态分配时该值为0或省略不写

		// 参数stream是一个cudaStream_t类型的可选参数，初始值为零，表示该核函数处在哪个流之中。
		*/

		if (nblocks == 4)
			compute_hists_kernel_many_blocks<4> << <grid, threads, smem, stream >> >(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
		else if (nblocks == 3)
			compute_hists_kernel_many_blocks<3> << <grid, threads, smem, stream >> >(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
		else if (nblocks == 2)
			compute_hists_kernel_many_blocks<2> << <grid, threads, smem, stream >> >(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);
		else
			compute_hists_kernel_many_blocks<1> << <grid, threads, smem, stream >> >(img_block_width, grad, qangle, scale, block_hists, cell_size_x, patch_size, block_patch_size, threads_cell, threads_block, half_cell_size);

		cudaSafeCall(cudaGetLastError());
	}


	//-------------------------------------------------------------
	//  通过L2Hys_norm(Lowe-style被截去的L2范数)对直方图进行归一化
	//


	// 归约共享内存
	template<int size>
	__device__ float reduce_smem(float* smem, float val)
	{
		unsigned int tid = threadIdx.x;
		float sum = val;
		/*
		// reduce函数作用：归约
		*/
		cv::cuda::device::reduce<size>(smem, sum, tid, cv::cuda::device::plus<float>());

		if (size == 32)
		{
#if __CUDA_ARCH__ >= 300
			return shfl(sum, 0);
#else
			return smem[0];
#endif
		}
		else
		{
#if __CUDA_ARCH__ >= 300
			if (threadIdx.x == 0)
				smem[0] = sum;
#endif

			__syncthreads();

			return smem[0];
		}
	}


	template <int nthreads, // 处理一个块直方图的线程数
		int nblocks> // 由一个GPU block处理的块直方图的数量
		__global__ void normalize_hists_kernel_many_blocks(const int block_hist_size,
		const int img_block_width,
		float* block_hists, float threshold)
	{
		if (blockIdx.x * blockDim.z + threadIdx.z >= img_block_width)
			return;

		float* hist = block_hists + (blockIdx.y * img_block_width +
			blockIdx.x * blockDim.z + threadIdx.z) *
			block_hist_size + threadIdx.x;

		__shared__ float sh_squares[nthreads * nblocks];
		float* squares = sh_squares + threadIdx.z * nthreads;

		float elem = 0.f;
		if (threadIdx.x < block_hist_size)
			elem = hist[0];

		__syncthreads(); // 预防竞争状况（多余？）
		float sum = reduce_smem<nthreads>(squares, elem * elem); // 平方和

		// 获得变换系数，及最大阈值（L2HysThreshold）
		float scale = 1.0f / (::sqrtf(sum) + 0.1f * block_hist_size);
		elem = ::min(elem * scale, threshold);

		__syncthreads(); // 预防竞争状况
		sum = reduce_smem<nthreads>(squares, elem * elem); // 在第一次的基础上继续求解平方和

		scale = 1.0f / (::sqrtf(sum) + 1e-3f);

		if (threadIdx.x < block_hist_size)
			hist[0] = elem * scale; // 直接乘以系数，得到最终的归一化结果  
	}


	void normalize_hists(int nbins,
		int block_stride_x, int block_stride_y,
		int height, int width,
		float* block_hists,
		float threshold,
		int cell_size_x, int cell_size_y,
		int ncells_block_x, int ncells_block_y,
		const cudaStream_t& stream)
	{
		const int nblocks = 1;

		// block_hist_size表示每个block直方图维数 
		int block_hist_size = nbins * ncells_block_x * ncells_block_y; // 9*2*2=36
		int nthreads = power_2up(block_hist_size);
		dim3 threads(nthreads, 1, nblocks); // (64,1,1)

		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;		// 15
		int img_block_height = (height - ncells_block_y * cell_size_y + block_stride_y) / block_stride_y;	// 7
		dim3 grid(cv::cuda::device::divUp(img_block_width, nblocks), img_block_height);						// (15,7)

		if (nthreads == 32)
			normalize_hists_kernel_many_blocks<32, nblocks> << <grid, threads, 0, stream >> >(block_hist_size, img_block_width, block_hists, threshold);
		else if (nthreads == 64)
			normalize_hists_kernel_many_blocks<64, nblocks> << <grid, threads, 0, stream >> >(block_hist_size, img_block_width, block_hists, threshold);
		else if (nthreads == 128)
			normalize_hists_kernel_many_blocks<128, nblocks> << <grid, threads, 0, stream >> >(block_hist_size, img_block_width, block_hists, threshold);
		else if (nthreads == 256)
			normalize_hists_kernel_many_blocks<256, nblocks> << <grid, threads, 0, stream >> >(block_hist_size, img_block_width, block_hists, threshold);
		else if (nthreads == 512)
			normalize_hists_kernel_many_blocks<512, nblocks> << <grid, threads, 0, stream >> >(block_hist_size, img_block_width, block_hists, threshold);
		else
			CV_Error(cv::Error::StsBadArg, "normalize_hists: histogram's size is too big, try to decrease number of bins");

		cudaSafeCall(cudaGetLastError());
	}


	//---------------------------------------------------------------------
	//  基于线性SVM的分类
	//

	// 返回置信度
	template <int nthreads, // Number of threads per one histogram block
		int nblocks>  // Number of histogram block processed by single GPU thread block
		__global__ void compute_confidence_hists_kernel_many_blocks(const int img_win_width, const int img_block_width,
		const int win_block_stride_x, const int win_block_stride_y,
		const float* block_hists, const float* coefs,
		float free_coef, float threshold, float* confidences)
	{
		const int win_x = threadIdx.z;
		if (blockIdx.x * blockDim.z + win_x >= img_win_width)
			return;

		const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
			blockIdx.x * win_block_stride_x * blockDim.z + win_x) *
			cblock_hist_size;

		float product = 0.f;
		for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
		{
			int offset_y = i / cdescr_width;
			int offset_x = i - offset_y * cdescr_width;
			product += coefs[i] * hist[offset_y * img_block_width * cblock_hist_size + offset_x];
		}

		__shared__ float products[nthreads * nblocks];

		const int tid = threadIdx.z * nthreads + threadIdx.x;

		cv::cuda::device::reduce<nthreads>(products, product, tid, cv::cuda::device::plus<float>());

		if (threadIdx.x == 0)
			confidences[blockIdx.y * img_win_width + blockIdx.x * blockDim.z + win_x] = product + free_coef;

	}

	void compute_confidence_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
		float* coefs, float free_coef, float threshold, int cell_size_x, int ncells_block_x, float *confidences)
	{
		const int nthreads = 256;
		const int nblocks = 1;

		int win_block_stride_x = win_stride_x / block_stride_x;
		int win_block_stride_y = win_stride_y / block_stride_y;
		int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
		int img_win_height = (height - win_height + win_stride_y) / win_stride_y;

		dim3 threads(nthreads, 1, nblocks);
		dim3 grid(cv::cuda::device::divUp(img_win_width, nblocks), img_win_height);

		cudaSafeCall(cudaFuncSetCacheConfig(compute_confidence_hists_kernel_many_blocks<nthreads, nblocks>,
			cudaFuncCachePreferL1));

		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) /
			block_stride_x;
		compute_confidence_hists_kernel_many_blocks<nthreads, nblocks> << <grid, threads >> >(
			img_win_width, img_block_width, win_block_stride_x, win_block_stride_y,
			block_hists, coefs, free_coef, threshold, confidences);
		cudaSafeCall(cudaThreadSynchronize());
	}



	template <int nthreads, // Number of threads per one histogram block
		int nblocks>  // Number of histogram block processed by single GPU thread block
		__global__ void classify_hists_kernel_many_blocks(const int img_win_width, const int img_block_width,
		const int win_block_stride_x, const int win_block_stride_y,
		const float* block_hists, const float* coefs,
		float free_coef, float threshold, unsigned char* labels)
	{
		const int win_x = threadIdx.z;
		if (blockIdx.x * blockDim.z + win_x >= img_win_width)
			return;

		const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
			blockIdx.x * win_block_stride_x * blockDim.z + win_x) *
			cblock_hist_size;

		float product = 0.f;
		for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
		{
			int offset_y = i / cdescr_width;
			int offset_x = i - offset_y * cdescr_width;
			product += coefs[i] * hist[offset_y * img_block_width * cblock_hist_size + offset_x];
		}

		__shared__ float products[nthreads * nblocks];

		const int tid = threadIdx.z * nthreads + threadIdx.x;

		cv::cuda::device::reduce<nthreads>(products, product, tid, cv::cuda::device::plus<float>());

		if (threadIdx.x == 0)
			labels[blockIdx.y * img_win_width + blockIdx.x * blockDim.z + win_x] = (product + free_coef >= threshold);
	}


	void classify_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
		float* coefs, float free_coef, float threshold, int cell_size_x, int ncells_block_x, unsigned char* labels)
	{
		const int nthreads = 256;
		const int nblocks = 1;

		int win_block_stride_x = win_stride_x / block_stride_x;
		int win_block_stride_y = win_stride_y / block_stride_y;
		int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
		int img_win_height = (height - win_height + win_stride_y) / win_stride_y;

		dim3 threads(nthreads, 1, nblocks);
		dim3 grid(cv::cuda::device::divUp(img_win_width, nblocks), img_win_height);

		cudaSafeCall(cudaFuncSetCacheConfig(classify_hists_kernel_many_blocks<nthreads, nblocks>, cudaFuncCachePreferL1));

		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;
		classify_hists_kernel_many_blocks<nthreads, nblocks> << <grid, threads >> >(
			img_win_width, img_block_width, win_block_stride_x, win_block_stride_y,
			block_hists, coefs, free_coef, threshold, labels);
		cudaSafeCall(cudaGetLastError());

		cudaSafeCall(cudaDeviceSynchronize());
	}

	//----------------------------------------------------------------------------
	// 提取descriptors


	template <int nthreads>
	__global__ void extract_descrs_by_rows_kernel(const int img_block_width,
		const int win_block_stride_x, const int win_block_stride_y,
		const float* block_hists,
		cv::cuda::PtrStepf descriptors)
	{
		// Get left top corner of the window in src
		const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
			blockIdx.x * win_block_stride_x) * cblock_hist_size;

		// Get left top corner of the window in dst
		float* descriptor = descriptors.ptr(blockIdx.y * gridDim.x + blockIdx.x);

		// Copy elements from src to dst
		for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
		{
			int offset_y = i / cdescr_width;
			int offset_x = i - offset_y * cdescr_width;
			descriptor[i] = hist[offset_y * img_block_width * cblock_hist_size + offset_x];
		}
	}


	void extract_descrs_by_rows(int win_height, int win_width,
		int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x,
		int height, int width,
		float* block_hists, int cell_size_x,
		int ncells_block_x,
		cv::cuda::PtrStepSzf descriptors,
		const cudaStream_t& stream)
	{
		// 为什么不是1024？
		const int nthreads = 256;

		int win_block_stride_x = win_stride_x / block_stride_x;
		int win_block_stride_y = win_stride_y / block_stride_y;
		int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
		int img_win_height = (height - win_height + win_stride_y) / win_stride_y;
		dim3 threads(nthreads, 1);
		dim3 grid(img_win_width, img_win_height);

		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;
		extract_descrs_by_rows_kernel<nthreads> << <grid, threads, 0, stream >> >(img_block_width, win_block_stride_x, win_block_stride_y, block_hists, descriptors);

		cudaSafeCall(cudaGetLastError());
	}


	template <int nthreads>
	__global__ void extract_descrs_by_cols_kernel(const int img_block_width,
		const int win_block_stride_x, const int win_block_stride_y,
		const float* block_hists,
		cv::cuda::PtrStepf descriptors)
	{
		// Get left top corner of the window in src
		const float* hist = block_hists + (blockIdx.y * win_block_stride_y * img_block_width +
			blockIdx.x * win_block_stride_x) * cblock_hist_size;

		// Get left top corner of the window in dst
		float* descriptor = descriptors.ptr(blockIdx.y * gridDim.x + blockIdx.x);

		// Copy elements from src to dst
		for (int i = threadIdx.x; i < cdescr_size; i += nthreads)
		{
			int block_idx = i / cblock_hist_size;
			int idx_in_block = i - block_idx * cblock_hist_size;

			int y = block_idx / cnblocks_win_x;
			int x = block_idx - y * cnblocks_win_x;

			descriptor[(x * cnblocks_win_y + y) * cblock_hist_size + idx_in_block]
				= hist[(y * img_block_width + x) * cblock_hist_size + idx_in_block];
		}
	}


	void extract_descrs_by_cols(int win_height, int win_width,
		int block_stride_y, int block_stride_x,
		int win_stride_y, int win_stride_x,
		int height, int width,
		float* block_hists,
		int cell_size_x, int ncells_block_x,
		cv::cuda::PtrStepSzf descriptors,
		const cudaStream_t& stream)
	{
		const int nthreads = 256;

		int win_block_stride_x = win_stride_x / block_stride_x;
		int win_block_stride_y = win_stride_y / block_stride_y;
		int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
		int img_win_height = (height - win_height + win_stride_y) / win_stride_y;
		dim3 threads(nthreads, 1);
		dim3 grid(img_win_width, img_win_height);

		int img_block_width = (width - ncells_block_x * cell_size_x + block_stride_x) / block_stride_x;
		extract_descrs_by_cols_kernel<nthreads> << <grid, threads, 0, stream >> >(img_block_width, win_block_stride_x, win_block_stride_y, block_hists, descriptors);

		cudaSafeCall(cudaGetLastError());
	}

	//----------------------------------------------------------------------------
	// Gradients computation


	template <int nthreads, int correct_gamma>
	__global__ void compute_gradients_8UC4_kernel(int height, int width, const cv::cuda::PtrStepb img,
		float angle_scale, cv::cuda::PtrStepf grad, cv::cuda::PtrStepb qangle)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;

		const uchar4* row = (const uchar4*)img.ptr(blockIdx.y);

		__shared__ float sh_row[(nthreads + 2) * 3];

		uchar4 val;
		if (x < width)
			val = row[x];
		else
			val = row[width - 2];

		sh_row[threadIdx.x + 1] = val.x;
		sh_row[threadIdx.x + 1 + (nthreads + 2)] = val.y;
		sh_row[threadIdx.x + 1 + 2 * (nthreads + 2)] = val.z;

		if (threadIdx.x == 0)
		{
			val = row[::max(x - 1, 1)];
			sh_row[0] = val.x;
			sh_row[(nthreads + 2)] = val.y;
			sh_row[2 * (nthreads + 2)] = val.z;
		}

		if (threadIdx.x == blockDim.x - 1)
		{
			val = row[::min(x + 1, width - 2)];
			sh_row[blockDim.x + 1] = val.x;
			sh_row[blockDim.x + 1 + (nthreads + 2)] = val.y;
			sh_row[blockDim.x + 1 + 2 * (nthreads + 2)] = val.z;
		}

		__syncthreads();
		if (x < width)
		{
			float3 a, b;

			b.x = sh_row[threadIdx.x + 2];
			b.y = sh_row[threadIdx.x + 2 + (nthreads + 2)];
			b.z = sh_row[threadIdx.x + 2 + 2 * (nthreads + 2)];
			a.x = sh_row[threadIdx.x];
			a.y = sh_row[threadIdx.x + (nthreads + 2)];
			a.z = sh_row[threadIdx.x + 2 * (nthreads + 2)];

			float3 dx;
			if (correct_gamma)
				dx = make_float3(::sqrtf(b.x) - ::sqrtf(a.x), ::sqrtf(b.y) - ::sqrtf(a.y), ::sqrtf(b.z) - ::sqrtf(a.z));
			else
				dx = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);

			float3 dy = make_float3(0.f, 0.f, 0.f);

			if (blockIdx.y > 0 && blockIdx.y < height - 1)
			{
				val = ((const uchar4*)img.ptr(blockIdx.y - 1))[x];
				a = make_float3(val.x, val.y, val.z);

				val = ((const uchar4*)img.ptr(blockIdx.y + 1))[x];
				b = make_float3(val.x, val.y, val.z);

				if (correct_gamma)
					dy = make_float3(::sqrtf(b.x) - ::sqrtf(a.x), ::sqrtf(b.y) - ::sqrtf(a.y), ::sqrtf(b.z) - ::sqrtf(a.z));
				else
					dy = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
			}

			float best_dx = dx.x;
			float best_dy = dy.x;

			float mag0 = dx.x * dx.x + dy.x * dy.x;
			float mag1 = dx.y * dx.y + dy.y * dy.y;
			if (mag0 < mag1)
			{
				best_dx = dx.y;
				best_dy = dy.y;
				mag0 = mag1;
			}

			mag1 = dx.z * dx.z + dy.z * dy.z;
			if (mag0 < mag1)
			{
				best_dx = dx.z;
				best_dy = dy.z;
				mag0 = mag1;
			}

			mag0 = ::sqrtf(mag0);

			float ang = (::atan2f(best_dy, best_dx) + CV_PI_F) * angle_scale - 0.5f;
			int hidx = (int)::floorf(ang);
			ang -= hidx;
			hidx = (hidx + cnbins) % cnbins;

			((uchar2*)qangle.ptr(blockIdx.y))[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
			((float2*)grad.ptr(blockIdx.y))[x] = make_float2(mag0 * (1.f - ang), mag0 * ang);
		}
	}


	void compute_gradients_8UC4(int nbins,
		int height, int width, const cv::cuda::PtrStepSzb& img,
		float angle_scale,
		cv::cuda::PtrStepSzf grad, cv::cuda::PtrStepSzb qangle,
		bool correct_gamma,
		const cudaStream_t& stream)
	{
		(void)nbins;
		const int nthreads = 256;

		dim3 bdim(nthreads, 1);
		dim3 gdim(cv::cuda::device::divUp(width, bdim.x), cv::cuda::device::divUp(height, bdim.y));

		if (correct_gamma)
			compute_gradients_8UC4_kernel<nthreads, 1> << <gdim, bdim, 0, stream >> >(height, width, img, angle_scale, grad, qangle);
		else
			compute_gradients_8UC4_kernel<nthreads, 0> << <gdim, bdim, 0, stream >> >(height, width, img, angle_scale, grad, qangle);

		cudaSafeCall(cudaGetLastError());
	}

	template <int nthreads, int correct_gamma>
	__global__ void compute_gradients_8UC1_kernel(int height, int width, const cv::cuda::PtrStepb img,
		float angle_scale, cv::cuda::PtrStepf grad, cv::cuda::PtrStepb qangle)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;

		const unsigned char* row = (const unsigned char*)img.ptr(blockIdx.y);

		__shared__ float sh_row[nthreads + 2];

		if (x < width)
			sh_row[threadIdx.x + 1] = row[x];
		else
			sh_row[threadIdx.x + 1] = row[width - 2];

		if (threadIdx.x == 0)
			sh_row[0] = row[::max(x - 1, 1)];

		if (threadIdx.x == blockDim.x - 1)
			sh_row[blockDim.x + 1] = row[::min(x + 1, width - 2)];

		__syncthreads();
		if (x < width)
		{
			float dx;

			if (correct_gamma)
				dx = ::sqrtf(sh_row[threadIdx.x + 2]) - ::sqrtf(sh_row[threadIdx.x]);
			else
				dx = sh_row[threadIdx.x + 2] - sh_row[threadIdx.x];

			float dy = 0.f;
			if (blockIdx.y > 0 && blockIdx.y < height - 1)
			{
				float a = ((const unsigned char*)img.ptr(blockIdx.y + 1))[x];
				float b = ((const unsigned char*)img.ptr(blockIdx.y - 1))[x];
				if (correct_gamma)
					dy = ::sqrtf(a) - ::sqrtf(b);
				else
					dy = a - b;
			}
			float mag = ::sqrtf(dx * dx + dy * dy);

			float ang = (::atan2f(dy, dx) + CV_PI_F) * angle_scale - 0.5f;
			int hidx = (int)::floorf(ang);
			ang -= hidx;
			hidx = (hidx + cnbins) % cnbins;

			((uchar2*)qangle.ptr(blockIdx.y))[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
			((float2*)grad.ptr(blockIdx.y))[x] = make_float2(mag * (1.f - ang), mag * ang);
		}
	}


	void compute_gradients_8UC1(int nbins,
		int height, int width, const cv::cuda::PtrStepSzb& img,
		float angle_scale,
		cv::cuda::PtrStepSzf grad, cv::cuda::PtrStepSzb qangle,
		bool correct_gamma,
		const cudaStream_t& stream)
	{
		(void)nbins;
		const int nthreads = 256;

		dim3 bdim(nthreads, 1);
		dim3 gdim(cv::cuda::device::divUp(width, bdim.x), cv::cuda::device::divUp(height, bdim.y));

		if (correct_gamma)
			compute_gradients_8UC1_kernel<nthreads, 1> << <gdim, bdim, 0, stream >> >(height, width, img, angle_scale, grad, qangle);
		else
			compute_gradients_8UC1_kernel<nthreads, 0> << <gdim, bdim, 0, stream >> >(height, width, img, angle_scale, grad, qangle);

		cudaSafeCall(cudaGetLastError());
	}



	/*
	//-------------------------------------------------------------------
	// Resize


	// 纹理内存是另一种类型的只读内存，在特定的访问模式中，纹理内存同样能够提升性能并减少内存流量
	// 纹理内存缓存在芯片上，因此在某些情况中，它能够减少对内存的请求并提供更高效的内存带宽
	// 纹理缓存是专门为那些在内存访问模式中存在大量空间局部性(Spatial Locality)的图形应用程序而设计的
	// 在某个计算应用程序中，这意味着一个线程读取的位置可能与邻近线程的读取位置“非常接近”
	*/

	// 读取纹理作为标准化的float
	texture<uchar4, 2, cudaReadModeNormalizedFloat> resize8UC4_tex;
	texture<uchar, 2, cudaReadModeNormalizedFloat> resize8UC1_tex;

	__global__ void resize_for_hog_kernel(float sx, float sy, cv::cuda::PtrStepSz<uchar> dst, int colOfs)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < dst.cols && y < dst.rows)
			dst.ptr(y)[x] = tex2D(resize8UC1_tex, x * sx + colOfs, y * sy) * 255;
	}

	__global__ void resize_for_hog_kernel(float sx, float sy, cv::cuda::PtrStepSz<uchar4> dst, int colOfs)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < dst.cols && y < dst.rows)
		{
			float4 val = tex2D(resize8UC4_tex, x * sx + colOfs, y * sy);
			dst.ptr(y)[x] = make_uchar4(val.x * 255, val.y * 255, val.z * 255, val.w * 255);
		}
	}

	template<class T, class TEX>
	static void resize_for_hog(const cv::cuda::PtrStepSzb& src, cv::cuda::PtrStepSzb dst, TEX& tex)
	{
		tex.filterMode = cudaFilterModeLinear;

		size_t texOfs = 0;
		int colOfs = 0;

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaSafeCall(cudaBindTexture2D(&texOfs, tex, src.data, desc, src.cols, src.rows, src.step));

		if (texOfs != 0)
		{
			colOfs = static_cast<int>(texOfs / sizeof(T));
			cudaSafeCall(cudaUnbindTexture(tex));
			cudaSafeCall(cudaBindTexture2D(&texOfs, tex, src.data, desc, src.cols, src.rows, src.step));
		}

		dim3 threads(32, 8);
		dim3 grid(cv::cuda::device::divUp(dst.cols, threads.x), cv::cuda::device::divUp(dst.rows, threads.y));

		float sx = static_cast<float>(src.cols) / dst.cols;
		float sy = static_cast<float>(src.rows) / dst.rows;

		resize_for_hog_kernel << <grid, threads >> >(sx, sy, (cv::cuda::PtrStepSz<T>)dst, colOfs);
		cudaSafeCall(cudaGetLastError());

		cudaSafeCall(cudaDeviceSynchronize());

		cudaSafeCall(cudaUnbindTexture(tex));
	}

	/*
	CUDA里的图像拉伸有很多方法，比较常用的方法是将图像放大后，利用卷积对图像进行滤波处理。这种方法时间复杂度高，处理复杂。
	相对而言，使用纹理内存进行图像拉伸，由于纹理内存本身对传入的图像数组有线性滤波作用，所以使用纹理拾取函数从源图像纹理内存取值，赋值给输出图像，即可完成图像拉伸及图像滤波两种功能。

	利用纹理内存的硬件插值功能，直接使用浮点型的坐标读取相应的源图像“像素”值，并赋值给目标图像。
	这里没有进行对源图像读取的越界检查，这是因为纹理内存硬件插值功能可以处理越界访问的情况，越界访问会按照事先的设置得到一个相对合理的像素颜色值，不会引起错误。
	使用cuda数组储存图像数据时，滤波效果更好。
	但是texture纹理内存做数据寄存器是不可取的，实验证明其存取速度并不高。纹理内存更适于图像放缩及图像旋转处理。
	*/
	void resize_8UC1(const cv::cuda::PtrStepSzb& src, cv::cuda::PtrStepSzb dst) { resize_for_hog<uchar>(src, dst, resize8UC1_tex); }
	void resize_8UC4(const cv::cuda::PtrStepSzb& src, cv::cuda::PtrStepSzb dst) { resize_for_hog<uchar4>(src, dst, resize8UC4_tex); }

	/* texture插值的问题
	// http://bbs.gpuworld.cn/thread-10385-1-1.html
	*/




}