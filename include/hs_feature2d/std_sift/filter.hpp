#ifndef _HS_FEATURE2D_STD_SIFT_FILTER_HPP_
#define _HS_FEATURE2D_STD_SIFT_FILTER_HPP_

#include <string>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <emmintrin.h>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_image_io/whole_io/image_io.hpp"
#include "hs_feature2d/std_sift/base_type.hpp"
#include "hs_feature2d/std_sift/ArrayHelper.hpp"


namespace hs{
namespace feature2d{


//尺寸对齐方法
static inline size_t alignSize(size_t sz, int n)
{
	assert((n & (n - 1)) == 0); // n 必须是2的整数倍
	return (sz + n - 1) & -n;
}

//指针对齐
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
	return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

//分离形式高斯滤波的行缓存管理类, 以16字节对齐方式分配内存
template<typename T>
struct RingBuffer
{
	//注意对于多通道图像, 行尺寸rw=像素数*通道数
	RingBuffer(size_t rw, size_t lines) : rowWidth(rw>0 ? rw : 0), 
		rowWidthA(rw>0 ? alignSize(rw, (HS_BYTE_ALIGN / sizeof(T))): 0),
		buffLines(lines)
	{
		if (buffLines > 0)
			linePtr.resize(buffLines);

		buffSize = buffLines * rowWidthA * sizeof(T);
		if (buffSize > 0)
		{
			buffer = (T*)_aligned_malloc(buffSize, HS_BYTE_ALIGN);
			for (int i = 0; i < buffLines; i++)
			{
				linePtr[i] = buffer + i * rowWidthA;
			}
			head = 0, tail = buffLines - 1;
		}
		else
			buffer = NULL;
	};

	~RingBuffer()
	{
		if (buffer != NULL)
			_aligned_free(buffer);
	};

	inline int Shift(){
		head = (head + 1) % buffLines; 
		offsetH++;
		tail = (head + buffLines - 1) % buffLines;
		return tail;
	};

	inline int Unshift(){
		head = (head - 1 + buffLines) % buffLines;
		offsetH--;
		tail = (head + buffLines - 1) % buffLines;
		return tail;
	};

	inline int First() { return head; }
	inline int Last(){ return tail; }
	inline void Next(int& i){ return (i + 1) % buffLines; }
	forceinline T* Line(int i){ return linePtr[ i % buffLines ]; }
	inline T* OffsetLine(int i){ return linePtr[(i - head) % buffLines]; }
	inline T* LastLine() { return linePtr[tail]; }
	inline T* FirstLine() { return linePtr[head]; }
	
	inline size_t BuffLines(){ return buffLines; }
private:
	size_t buffSize, rowWidth, rowWidthA, buffLines, offsetH = 0;
	int head, tail;
	std::vector< T* > linePtr;
	T* buffer;
};

//通用行滤波functor
template<typename ST, typename DT>
struct RowFilter
{
	void operator()(ST* srcbuff, DT* dst, float* mask_ptr, int& kw, int& w, int& cn)
	{
		int row_size = w * cn, j, k;
		//初始化目标行元素
		memset(dst, 0, sizeof(DT) * row_size);
		//遍历行模板
		for (k = 0; k < kw * 2 + 1; k++, srcbuff += cn)
		{
			//遍历行元素
			j = 0;
#if HS_ENABLE_UNROLLED
			for (; j <= row_size - 4; j += 4)
			{
				dst[j] += DT(srcbuff[j] * mask_ptr[k]);
				dst[j + 1] += DT(srcbuff[j + 1] * mask_ptr[k]);
				dst[j + 2] += DT(srcbuff[j + 2] * mask_ptr[k]);
				dst[j + 3] += DT(srcbuff[j + 3] * mask_ptr[k]);
			}
#endif
			for (; j < row_size; j++)
			{
				dst[j] += DT(srcbuff[j] * mask_ptr[k]);
			}
		}
	}
};

//针对float类型的SSE加速版本
template<> struct RowFilter<float, float>
{
	forceinline void operator()(float* srcbuff, float* dst, float* mask_ptr, int& kw, int& w, int& cn)
	{
#ifdef  _USE_SSE2_
		int row_width = w*cn, j, k, _ksize = kw * 2 + 1;
		float* src;
		__m128 f, s0, s1, s0_, s1_, x0, x1, x0_, x1_;
		j = 0;
#if HS_ENABLE_UNROLLED
		for (; j <= row_width - 16; j += 16)
		{
			src = srcbuff + j;
			s0 = s1 = s0_ = s1_ = _mm_setzero_ps();
			for (k = 0; k < _ksize; k++, src += cn)
			{
				f = _mm_load_ss(mask_ptr + k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_loadu_ps(src);
				x1 = _mm_loadu_ps(src + 4);
				s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));

				x0_ = _mm_loadu_ps(src + 8);
				x1_ = _mm_loadu_ps(src + 12);
				s0_ = _mm_add_ps(s0_, _mm_mul_ps(x0_, f));
				s1_ = _mm_add_ps(s1_, _mm_mul_ps(x1_, f));
			}
			_mm_store_ps(dst + j, s0);
			_mm_store_ps(dst + j + 4, s1);

			_mm_store_ps(dst + j + 8, s0_);
			_mm_store_ps(dst + j + 12, s1_);
		}
#endif
		for (; j <= row_width - 8; j += 8)
		{
			src = srcbuff + j;
			s0 = s1 = _mm_setzero_ps();
			for (k = 0; k < _ksize; k++, src += cn)
			{
				//将单个32f的模板元素阵列至128位
				f = _mm_load_ss(mask_ptr + k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_loadu_ps(src);
				x1 = _mm_loadu_ps(src + 4);
				s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
			}
			_mm_store_ps(dst + j, s0);
			_mm_store_ps(dst + j + 4, s1);
		}

		for (; j <= row_width - 4; j += 4)
		{
			src = srcbuff + j;
			__m128 f, s0 = _mm_setzero_ps(), x0;
			for (k = 0; k < _ksize; k++, src += cn)
			{
				f = _mm_load_ss(mask_ptr + k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_loadu_ps(src);
				s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
			}
			_mm_store_ps(dst + j, s0);
		}

		for (; j < row_width; j++)
		{
			src = srcbuff + j;
			dst[j] *= 0;
			for (k = 0; k < _ksize; k++, src += cn)
			{
				dst[j] += (*src) * mask_ptr[k];
			}
		}
#else
		int row_size = w * cn, i, j, k, _ksize = kw * 2 + 1;
		float* src;
		//初始化目标行元素
		memset(dst, 0, sizeof(float) * row_size);
		for (k = 0; k < kw * 2 + 1; k++, srcbuff += cn)
		{
			//遍历行元素
			j = 0;
#if HS_ENABLE_UNROLLED
			for (; j <= row_size - 4; j += 4)
			{
				dst[j] += srcbuff[j] * mask_ptr[k];
				dst[j + 1] += srcbuff[j + 1] * mask_ptr[k];
				dst[j + 2] += srcbuff[j + 2] * mask_ptr[k];
				dst[j + 3] += srcbuff[j + 3] * mask_ptr[k];
			}
#endif
			for (; j < row_size; j++)
			{
				dst[j] += srcbuff[j] * mask_ptr[k];
			}
		}
#endif // _USE_SSE2_
	}
};

//通用的列滤波functor
template<typename ST, typename DT>
struct ColFilter
{
	inline void operator()(RingBuffer<ST>& ringbuff, DT* dst, float* mask_ptr_h, int& idx, int& height_, int& w, int& h, int& cn)
	{
		int row_width = w * cn, j, k, c;
		ST *ptmp;
		memset(dst, 0, row_width * sizeof(DT));
		for (k = 0; k < height_ * 2 + 1; k++)
		{
			c = idx + k - height_;
			c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			ptmp = ringbuff.Line(c);
			j = 0;
#if HS_ENABLE_UNROLLED
			for (; j <= row_width - 4; j+=4)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
				dst[j+1] += mask_ptr_h[k] * ptmp[j+1];
				dst[j+2] += mask_ptr_h[k] * ptmp[j+2];
				dst[j+3] += mask_ptr_h[k] * ptmp[j+3];
			}
#endif
			for (; j < row_width; j++)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
			}
		}
	}
};

//针对float类型的SSE加速版本
template<> struct ColFilter< float, float>
{
	forceinline void operator()(RingBuffer<float>& ringbuff, float* dst, float* mask_ptr_h, int& idx, int& height_, int& w, int& h, int& cn)
	{
#ifdef  _USE_SSE2_
		int c, c0 = idx - height_, j, k, row_width = w * cn, r16 = row_width - 16, r8 = row_width - 8, r4 = row_width - 4;
		float *ptmp;
		__m128 f;
		__m128 s0, s0_, s1, s1_, s2, s2_, s3, s3_, x0, x1, x0_, x1_;

		for (k = 0; k < height_ * 2 + 1; k++)
		{
			c = c0 + k;
			c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			j = 0;
			ptmp = ringbuff.Line(c);
			f = _mm_load_ps(mask_ptr_h + k);
			f = _mm_shuffle_ps(f, f, 0);
			for (; j <= r16; j += 16)
			{

				x0 = _mm_load_ps(ptmp + j);
				x1 = _mm_load_ps(ptmp + j + 4);
				s0 = _mm_loadu_ps(dst + j);
				s1 = _mm_loadu_ps(dst + j + 4);
				s2 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
				_mm_storeu_ps(dst + j, s2);
				_mm_storeu_ps(dst + j + 4, s3);

				x0_ = _mm_load_ps(ptmp + j + 8);
				x1_ = _mm_load_ps(ptmp + j + 12);
				s0_ = _mm_loadu_ps(dst + j + 8);
				s1_ = _mm_loadu_ps(dst + j + 12);
				s2_ = _mm_add_ps(s0_, _mm_mul_ps(x0_, f));
				s3_ = _mm_add_ps(s1_, _mm_mul_ps(x1_, f));
				_mm_storeu_ps(dst + j + 8, s2_);
				_mm_storeu_ps(dst + j + 12, s3_);
			}
			for (; j <= r8; j += 8)
			{

				x0 = _mm_load_ps(ptmp + j);
				x1 = _mm_load_ps(ptmp + j + 4);
				s0 = _mm_loadu_ps(dst + j);
				s1 = _mm_loadu_ps(dst + j + 4);
				s2 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
				_mm_storeu_ps(dst + j, s2);
				_mm_storeu_ps(dst + j + 4, s3);
			}
			for (; j <= r4; j += 4)
			{

				x0 = _mm_load_ps(ptmp + j);
				s0 = _mm_loadu_ps(dst + j);
				s1 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				_mm_storeu_ps(dst + j, s1);
			}
			for (; j < row_width; j++)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
			}
		}
#else
		int row_width = w * cn, j, k, c;
		float *ptmp;
		memset(dst, 0, row_width * sizeof(float));
		for (k = 0; k < height_ * 2 + 1; k++)
		{
			c = idx + k - height_;
			c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			ptmp = ringbuff.Line(c);
			j = 0;
#if HS_ENABLE_UNROLLED
			for (; j <= row_width - 4; j+=4)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
				dst[j+1] += mask_ptr_h[k] * ptmp[j+1];
				dst[j+2] += mask_ptr_h[k] * ptmp[j+2];
				dst[j+3] += mask_ptr_h[k] * ptmp[j+3];
			}
#endif
			for (; j < row_width; j++)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
			}
		}
#endif
	}
};


//高斯滤波类
class HS_EXPORT GaussianFilter
{
public:
	typedef float MaskDataType;
	typedef HeapMgr<MaskDataType> T_MaskVector;
	enum GsError
	{
		CREATE_MASK_FAIL = -1,
		GS_OK = 0,
		INVALID_SOURCE,
		CREATE_IMG_FAIL,
		INVALID_CHANNEL
	};

	GaussianFilter(){ this->initMask(); };
	GaussianFilter(float s, int w = 0, int h = 0) : sigmaX_(s), sigmaY_(s)
	{
		width_ = (w > 0) ? (w) : std::ceil(3 * sigmaX_);
		height_ = (h > 0) ? (h) : std::ceil(3 * sigmaY_);
		this->initMask();
	};
	GaussianFilter(float sx, float sy, int w = 0, int h = 0) : sigmaX_(sx), sigmaY_(sy)
	{
		width_ = (w > 0) ? (w) : std::ceil(3 * sigmaX_);
		height_ = (h > 0) ? (h) : std::ceil(3 * sigmaY_);
		this->initMask();
	};
	~GaussianFilter(){};

	HeapMgr<float>& GetMask(int idx){
		return mask_w;
	};

	void SetMask(float sx, float sy = 0.0f, int w = 0, int h = 0);

	//主要操作入口
	template<typename ST, typename DT = float>
	int Apply(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
	{
		int res = 0;
		int w = img_input.width(), h = img_input.height(), cn = img_input.channel();
		int row_width = w * cn, row_step = w * cn * sizeof(DT);
		res = img_output.CreateImage(w, h, cn, sizeof(DT) * 8, img_input.color_type(), true);
		
		int i, j, k;
		int c = 0, startH = 0, _ksize = 2 * width_ + 1;
		float *mask_ptr, *mask_ptr_h;

		//镜像边界的元素索引
		HeapMgr<int> border_idx(width_ * 2);
		int *pbdr = border_idx.GetPtr();
		int border_offset = width_*cn;
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = (width_ - i)*cn, pbdr[i + width_] = row_width - (2 + i)*cn;
		}

		// 行滤波缓冲
		RingBuffer<DT> ringbuff(row_width, std::min(h * height_, h));
		//行滤波器
		RowFilter<ST, DT> rFilter;
		//列滤波器
		ColFilter<ST, DT> cFilter;
		
		//源与目标图像缓冲行
		HeapMgrA<ST> srcBuffRow(row_width + width_ * 2 * cn);
		ST *srcbuff, *src;
		DT *dst = NULL, *ptmp = NULL;

		//初始化目标图像
		dst = img_output.GetBufferT<float>();
		memset(dst, 0, row_width * h * sizeof(float));
		
		mask_ptr = mask_w.GetPtr(), mask_ptr_h = mask_h.GetPtr();
//#pragma omp parallel for
		for (i = 0; i < h; i++)
		{
			//设置源图像的缓冲行
			srcbuff = srcBuffRow.GetPtr();
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_width * sizeof(ST));
			//设置缓冲行的镜像边界
			switch (cn)
			{
			case 1:
				for (k = 0; k < width_; k++)
				{
					srcbuff[k * cn] = src[pbdr[k] * cn], srcbuff[(border_offset + w + k)*cn] = src[pbdr[k + width_] * cn];
				}
				break;
			case 3:
				for (k = 0; k < width_; k++)
				{
					srcbuff[k * cn] = src[pbdr[k] * cn], srcbuff[(border_offset + w + k)*cn] = src[pbdr[k + width_] * cn];
					srcbuff[k*cn + 1] = src[pbdr[k] * cn + 1], srcbuff[(border_offset + w + k)*cn + 1] = src[pbdr[k + width_] * cn + 1];
					srcbuff[k*cn + 2] = src[pbdr[k] * cn + 2], srcbuff[(border_offset + w + k)*cn + 2] = src[pbdr[k + width_] * cn + 2];
				}
				break;

			default:
				for (k = 0; k < width_; k++)
				{
					for (j = 0; j < cn; j++)
					{
						srcbuff[k*cn + j] = src[pbdr[k] * cn + j], srcbuff[(border_offset + w + k)*cn + j] = src[pbdr[k + width_] * cn + j];
					}
				}
				break;
			}
			
			//设置中间环形缓冲行指针
			if (i >= ringbuff.BuffLines()){
				ringbuff.Shift();
				dst = ringbuff.LastLine();
			}
			else
			{
				dst = ringbuff.Line(i);
			}
			//调用行滤波functor
			rFilter(srcbuff, dst, mask_ptr, width_, w, cn);

			if (i >= height_) //已缓冲足够的行, 可同步开始列滤波
			{ 
				startH = i - height_;
				dst = img_output.GetLineT<DT>(startH);
				//调用列滤波functor
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}
		}

		if (startH < h - 1)
		{
			//处理剩余的行
			for (i = startH; i < h; i++)
			{
				dst = img_output.GetLineT<DT>(i);
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}
		}
		return res;
	};

	template<>
	int Apply<float, float>(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
	{
		int res = 0;
		int w = img_input.width(), h = img_input.height(), cn = img_input.channel(), bits = img_input.bit_depth();
		int row_width = w * cn, row_step = w * cn * sizeof(float);
		if (w != img_output.width() || h != img_output.height() || cn != img_output.channel() || bits != img_output.bit_depth())
		{
			res = img_output.CreateImage(w, h, cn, bits, img_input.color_type(), true);
		}

		int i, j, k;
		int c = 0, startH = 0, _ksize = 2 * width_ + 1;
		float *mask_ptr, *mask_ptr_h;

		//镜像边界的元素索引
		HeapMgr<int> border_idx(width_ * 2);
		int *pbdr = border_idx.GetPtr();
		int border_offset = width_*cn;
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = (width_ - i)*cn, pbdr[i + width_] = row_width - (2 + i)*cn;
		}

		// 行滤波缓冲
		int iBuffLine = std::min(2 * height_ + 1, h);
		RingBuffer<float> ringbuff(row_width, iBuffLine);
		//行滤波器
		RowFilter<float, float> rFilter;
		//列滤波器
		ColFilter<float, float> cFilter;

		//源与目标图像缓冲行
		HeapMgrA<float> srcBuffRow(row_width + width_ * 2 * cn);
		float *srcbuff, *src;
		float *dst = NULL, *ptmp = NULL;

		//初始化目标图像
		dst = img_output.GetBufferT<float>();
		memset(dst, 0, row_width * h * sizeof(float));

		mask_ptr = mask_w.GetPtr(), mask_ptr_h = mask_h.GetPtr();
		//#pragma omp parallel for
		for (i = 0; i < h; i++)
		{
			//设置源图像的缓冲行
			srcbuff = srcBuffRow.GetPtr();
			src = (float*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_width * sizeof(float));
			//设置缓冲行的镜像边界
			switch (cn)
			{
			case 1:
				for (k = 0; k < width_; k++)
				{
					srcbuff[k * cn] = src[pbdr[k] * cn], srcbuff[(border_offset + w + k)*cn] = src[pbdr[k + width_] * cn];
				}
				break;
			case 3:
				for (k = 0; k < width_; k++)
				{
					srcbuff[k * cn] = src[pbdr[k] * cn], srcbuff[(border_offset + w + k)*cn] = src[pbdr[k + width_] * cn];
					srcbuff[k*cn + 1] = src[pbdr[k] * cn + 1], srcbuff[(border_offset + w + k)*cn + 1] = src[pbdr[k + width_] * cn + 1];
					srcbuff[k*cn + 2] = src[pbdr[k] * cn + 2], srcbuff[(border_offset + w + k)*cn + 2] = src[pbdr[k + width_] * cn + 2];
				}
				break;

			default:
				for (k = 0; k < width_; k++)
				{
					for (j = 0; j < cn; j++)
					{
						srcbuff[k*cn + j] = src[pbdr[k] * cn + j], srcbuff[(border_offset + w + k)*cn + j] = src[pbdr[k + width_] * cn + j];
					}
				}
				break;
			}

			//设置中间环形缓冲行指针
			if (i >= iBuffLine){
				ringbuff.Shift();
				dst = ringbuff.LastLine();
			}
			else
			{
				dst = ringbuff.Line(i);
			}
			//调用行滤波functor
			rFilter(srcbuff, dst, mask_ptr, width_, w, cn);

			if (i >= height_) //已缓冲足够的行, 可同步开始列滤波
			{
				startH = i - height_;
				dst = img_output.GetLineT<float>(startH);
				//调用列滤波functor
				cFilter(ringbuff, dst, mask_ptr_h, startH, height_, w, h, cn);
			}
		}

		if (startH < h - 1)
		{
			//处理剩余的行
			for (i = startH + 1; i < h; i++)
			{
				dst = img_output.GetLineT<float>(i);
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}
		}
		return res;
	};


private:
	void initMask();

	int state_;
	HeapMgr<float> mask_w, mask_h;
	HeapMgr<int> imask_w; //用于8位-32位sse优化的中间模板
	float sigmaX_ = 0.6, sigmaY_ = 0.6;
	float su_ = 0, sv_ = 0; //归一化参数
	int width_ = 2, height_ = 2;
};




}
}

#endif //_HS_FEATURE2D_STD_SIFT_FILTER_HPP_