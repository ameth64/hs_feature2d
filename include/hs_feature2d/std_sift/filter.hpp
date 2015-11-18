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
	RingBuffer() : rowWidth(0), rowWidthA(0), buffLines(0), buffSize(0), buffer(NULL)
	{ };

	//注意对于多通道图像, 行尺寸rw=像素数*通道数
	RingBuffer(size_t rw, size_t lines) : rowWidth(rw>0 ? rw : 0), 
		rowWidthA(rw>0 ? alignSize(rw, (HS_BYTE_ALIGN / sizeof(T))): 0),
		buffLines(lines)
	{
		SetBuffer();
	};

	~RingBuffer()
	{
		if (buffer != NULL)
			_aligned_free(buffer);
	};

	inline void Initialize(size_t rw, size_t lines)
	{
		rowWidth = (rw > 0 ? rw : 0); 
		rowWidthA = rw > 0 ? alignSize(rw, (HS_BYTE_ALIGN / sizeof(T))) : 0;
		buffLines = lines;
		SetBuffer();
	}

	inline void SetBuffer()
	{
		if (buffLines > 0 && buffLines != linePtr.size())
			linePtr.resize(buffLines);
		head = 0, tail = buffLines - 1;
		size_t _buffSize = buffLines * rowWidthA * sizeof(T);
		if (_buffSize > 0 && _buffSize != buffSize)
		{
			buffSize = _buffSize;
			if (buffer != NULL)
				_aligned_free(buffer);
			buffer = (T*)_aligned_malloc(buffSize, HS_BYTE_ALIGN);
			for (int i = 0; i < buffLines; i++)
			{
				linePtr[i] = buffer + i * rowWidthA;
			}			
		}
		if (_buffSize == 0)
			buffer = NULL;
	};

	forceinline void Shift(){
		head = (head + 1) % buffLines; 
		offsetH++;
		tail = (head + buffLines - 1) % buffLines;
	};

	forceinline void Unshift(){
		head = (head - 1 + buffLines) % buffLines;
		offsetH--;
		tail = (head + buffLines - 1) % buffLines;
	};

	inline int First() { return head; }
	inline int Last(){ return tail; }
	inline void Next(int& i){ return (i + 1) % buffLines; }
	forceinline T* Line(int i){ return linePtr[i % buffLines]; }
	forceinline T* OffsetLine(int i){ return linePtr[(i - head) % buffLines]; }
	forceinline T* LastLine() { return linePtr[tail]; }
	forceinline T* FirstLine() { return linePtr[head]; }
	
	inline size_t BuffLines(){ return buffLines; }
	inline float** LinePtr(){ return &(linePtr[0]); }
private:
	size_t buffSize, rowWidth, rowWidthA, buffLines, offsetH = 0;
	int head, tail;
	std::vector< T* > linePtr;
	T* buffer;
};

//调试选项
#define HS_ROW_DEBUG 0
#define HS_COL_DEBUG 0

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
		const int row_width = w*cn, _ksize = kw * 2 + 1, r16 = row_width - 16, r8 = row_width - 8, r4 = row_width - 4;
		int j, k;
		float* src;
		__m128 f, s0, s1, s0_, s1_, x0, x1, x0_, x1_;
		j = 0;
#if HS_ENABLE_UNROLLED
		for (; j <= r16; j += 16)
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
		for (; j <= r8; j += 8)
		{
			src = srcbuff + j;
			s0 = s1 = _mm_setzero_ps();
			for (k = 0; k < _ksize; k++, src += cn)
			{
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

		for (; j <= r4; j += 4)
		{
			src = srcbuff + j;
			s0 = _mm_setzero_ps();
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
			dst[j] *= 0.0f;
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
	forceinline void operator()(float** ringbuff, float* dst, float* mask_ptr_h, int& idx, int& height_, int& w, int& cn)
	{
#ifdef  _USE_SSE2_
		int j, k;
		const int row_width = w * cn; 
		const int r16 = row_width - 16;
		const int r8 = row_width - 8;
		const int r4 = row_width - 4;
		float** rbptr = ringbuff + height_;
		float *ptmp, *ptmp2;
		__m128 f, f2;
		__m128 s0, s1, s2, s3, x0, x1;
		__m128 s0_, s1_, s2_, s3_, x0_, x1_;

		j = 0;
		for (; j <= r16; j += 16)
		{
			ptmp = rbptr[0] + j;
			f = _mm_load_ps(mask_ptr_h + height_);
			f = _mm_shuffle_ps(f, f, 0);

			x0 = _mm_load_ps(ptmp);
			x1 = _mm_load_ps(ptmp + 4);
			s2 = _mm_mul_ps(x0, f);
			s3 = _mm_mul_ps(x1, f);

			x0_ = _mm_load_ps(ptmp + 8);
			x1_ = _mm_load_ps(ptmp + 12);
			s2_ = _mm_mul_ps(x0_, f);
			s3_ = _mm_mul_ps(x1_, f);

			for (k = 1; k <= height_; k++)
			{
				ptmp = rbptr[k] + j;
				ptmp2 = rbptr[-k] + j;
				f = _mm_load_ps(mask_ptr_h + height_ - k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_load_ps(ptmp);
				x1 = _mm_load_ps(ptmp + 4);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
				x0_ = _mm_load_ps(ptmp + 8);
				x1_ = _mm_load_ps(ptmp + 12);
				s2_ = _mm_add_ps(s2_, _mm_mul_ps(x0_, f));
				s3_ = _mm_add_ps(s3_, _mm_mul_ps(x1_, f));

				x0 = _mm_load_ps(ptmp2);
				x1 = _mm_load_ps(ptmp2 + 4);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
				x0_ = _mm_load_ps(ptmp2 + 8);
				x1_ = _mm_load_ps(ptmp2 + 12);
				s2_ = _mm_add_ps(s2_, _mm_mul_ps(x0_, f));
				s3_ = _mm_add_ps(s3_, _mm_mul_ps(x1_, f));
			}
			_mm_storeu_ps(dst + j, s2);
			_mm_storeu_ps(dst + j + 4, s3);
			_mm_storeu_ps(dst + j + 8, s2_);
			_mm_storeu_ps(dst + j + 12, s3_);
		}
		for (; j <= r8; j+=8)
		{
			ptmp = rbptr[0] + j;
			f = _mm_load_ps(mask_ptr_h + height_);
			f = _mm_shuffle_ps(f, f, 0);
			x0 = _mm_load_ps(ptmp);
			x1 = _mm_load_ps(ptmp + 4);
			s2 = _mm_mul_ps(x0, f);
			s3 = _mm_mul_ps(x1, f);
			for (k = 1; k <= height_; k++)
			{
				ptmp = rbptr[k] + j;
				ptmp2 = rbptr[-k] + j;
				f = _mm_load_ps(mask_ptr_h + height_ - k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_load_ps(ptmp);
				x1 = _mm_load_ps(ptmp + 4);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));

				x0 = _mm_load_ps(ptmp2);
				x1 = _mm_load_ps(ptmp2 + 4);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
				s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
			}
			_mm_storeu_ps(dst + j, s2);
			_mm_storeu_ps(dst + j + 4, s3);
		}
		for (; j <= r4; j += 4)
		{
			ptmp = rbptr[0] + j;
			f = _mm_load_ps(mask_ptr_h + height_);
			f = _mm_shuffle_ps(f, f, 0);
			x0 = _mm_load_ps(ptmp);
			s2 = _mm_mul_ps(x0, f);
			for (k = 1; k <= height_; k++)
			{
				ptmp = rbptr[k] + j;
				ptmp2 = rbptr[-k] + j;
				f = _mm_load_ps(mask_ptr_h + height_ - k);
				f = _mm_shuffle_ps(f, f, 0);

				x0 = _mm_load_ps(ptmp);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));

				x0 = _mm_load_ps(ptmp2);
				s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
			}
			_mm_storeu_ps(dst + j, s2);
		}
		for (; j < row_width; j++)
		{
			ptmp = rbptr[0] + j;
			dst[j] = ptmp[j] * mask_ptr_h[height_];
			for (k = 1; k <= height_; k++)
			{
				ptmp = rbptr[k] + j;
				ptmp2 = rbptr[-k] + j;
				dst[j] += ptmp[0] * mask_ptr_h[height_ - k];
				dst[j] += ptmp2[0] * mask_ptr_h[height_ - k];
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
template<typename ST, typename DT >
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

	GaussianFilter(){};

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

	void SetMask(float sx, float sy = 0.0f, int w = 0, int h = 0)
	{
		sigmaX_ = (sx > FLT_EPSILON) ? sx : 0.6f;
		sigmaY_ = (sy > FLT_EPSILON) ? sy : sigmaX_;
		width_ = (w > 0) ? w : int(3 * sigmaX_);
		height_ = (h > 0) ? h : int(3 * sigmaY_);
		initMask();
	};

	//主要操作入口
	int Apply(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
	{
		int res = 0;
		int w = img_input.width(), h = img_input.height(), cn = img_input.channel(), bits = img_input.bit_depth();
		const int row_width = w * cn, row_step = w * cn * sizeof(ST);
		if (w != img_output.width() || h != img_output.height() || cn != img_output.channel() || bits != img_output.bit_depth())
		{
			res = img_output.CreateImage(w, h, cn, bits, img_input.color_type(), true);
		}
		
		int i, j, k, c = 0, startH = 0;

		border_offset = width_*cn;
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = width_ - i, pbdr[i + width_] = w - 2 - i;
		}

		// 行滤波缓冲
		ringbuff.Initialize(row_width, std::min(ksize_h, h));

		//源图像缓冲行
		srcBuffRow.resize(row_width + width_ * 2 * cn);

		//初始化目标图像
		dst = img_output.GetBufferT<DT>();
		
		srcbuff = &(srcBuffRow[0]);
		i = 0;
		for (; i < height_; i++)
		{
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_step);
			interpolateBorder(w, cn, k, j);
			dst = ringbuff.Line(i);
			rFilter(srcbuff, dst, mask_ptr_w, width_, w, cn);
		}
		for (; i < ksize_h; i++)
		{
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_step);
			interpolateBorder(w, cn, k, j);
			dst = ringbuff.Line(i);
			rFilter(srcbuff, dst, mask_ptr_w, width_, w, cn);


			startH = i - height_;
			dst = img_output.GetLineT<DT>(startH);
			for (j = 0; j < ksize_h; j++)
			{
				c = startH - height_ + j;
				//col_border_idx[j] = ringbuff.Line(c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c));
				col_border_idx[j] = ringbuff.Line(c < 0 ? -c : c);
			}
			cFilter(&(col_border_idx[0]), dst, mask_ptr_h, startH, height_, w, cn);
		}

		for (; i < h; i++)
		{
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_step);
			interpolateBorder(w, cn, k, j);
			//设置环形缓冲区起始索引位移
			ringbuff.Shift();
			dst = ringbuff.LastLine();

			//调用行滤波functor
			rFilter(srcbuff, dst, mask_ptr_w, width_, w, cn);

			startH = i - height_;
			dst = img_output.GetLineT<DT>(startH);

			for (j = 0; j < ksize_h; j++)
			{
				c = startH - height_ + j;
				//col_border_idx[j] = ringbuff.Line(c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c));
				col_border_idx[j] = ringbuff.Line(c);
			}
			cFilter(&(col_border_idx[0]), dst, mask_ptr_h, startH, height_, w, cn);
		}

		if (startH < h - 1)
		{
			//处理剩余的行
			for (i = startH+1; i < h; i++)
			{
				dst = img_output.GetLineT<DT>(i);
				for (j = 0; j < ksize_h; j++)
				{
					c = i - height_ + j;
					//col_border_idx[j] = ringbuff.Line(c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c));
					col_border_idx[j] = ringbuff.Line(c >= h ? 2 * h - c - 2 : c);
				}
				cFilter(&(col_border_idx[0]), dst, mask_ptr_h, startH, height_, w, cn);
			}
		}
		return res;
	};

private:
	void initMask()
	{
		ksize_w = width_ * 2 + 1, ksize_h = height_ * 2 + 1;
		bool res = mask_w.Allocate(ksize_w) && mask_h.Allocate(ksize_h) && imask_w.Allocate(ksize_w);
		if (res == false)
		{
			this->state_ = int(GaussianFilter::GsError::CREATE_MASK_FAIL);
		}
		else
			this->state_ = int(GaussianFilter::GsError::GS_OK);

		mask_ptr_w = mask_w.GetPtr(), mask_ptr_h = mask_h.GetPtr();
		float tmp = 0, cons = (2 * sigmaX_ * sigmaX_), *ptr = mask_w.GetPtr();
		int *iptr = imask_w.GetPtr();
		//计算行向量
		su_ = 0;
		for (int i = -width_; i <= width_; i++)
		{
			tmp = std::exp(-(i*i) / cons);
			ptr[i + width_] = tmp, su_ += tmp;
		}
		for (int i = -width_; i <= width_; i++)
		{
			ptr[i + width_] /= su_;
			iptr[i + width_] = int(ptr[i + width_] * 256);
		}
		//设置边界镜像索引
		border_idx.Allocate(width_ * 2);
		pbdr = border_idx.GetPtr();

		//设置列滤波镜像边界索引
		col_border_idx.resize(2 * height_ + 1);

		//计算列向量
		// 如果行列相等
		if (height_ == width_ && sigmaX_ == sigmaY_)
		{
			sv_ = su_;
			float *ptr2 = mask_h.GetPtr();
			for (int i = -width_; i <= width_; i++)
			{
				ptr2[i + width_] = ptr[i + width_];
			}
			return;
		}
		cons = (2 * sigmaY_ * sigmaY_);
		sv_ = 0;
		for (int i = -height_; i <= height_; i++)
		{
			tmp = std::exp(-(i*i) / cons);
			mask_h.GetPtr()[i + height_] = tmp, sv_ += tmp;
		}
		for (int i = -height_; i <= height_; i++)
		{
			mask_h.GetPtr()[i + height_] /= sv_;
		}

		return;
	};

	forceinline void interpolateBorder(const int& w, const int& cn, int& k, int& j)
	{
		k = 0;
		switch (cn)
		{
		case 1:
			for (; k < width_; k++)
			{
				srcbuff[k] = src[pbdr[k]], srcbuff[(width_ + w + k)] = src[pbdr[k + width_]];
			}
			break;
		case 3:
			for (; k < width_; k++)
			{
				srcbuff[k * cn] = src[pbdr[k] * cn], srcbuff[(width_ + w + k)*cn] = src[pbdr[k + width_] * cn];
				srcbuff[k*cn + 1] = src[pbdr[k] * cn + 1], srcbuff[(width_ + w + k)*cn + 1] = src[pbdr[k + width_] * cn + 1];
				srcbuff[k*cn + 2] = src[pbdr[k] * cn + 2], srcbuff[(width_ + w + k)*cn + 2] = src[pbdr[k + width_] * cn + 2];
			}
			break;

		default:
			for (; k < width_; k++)
			{
				for (j = 0; j < cn; j++)
				{
					srcbuff[k*cn + j] = src[pbdr[k] * cn + j], srcbuff[(width_ + w + k)*cn + j] = src[pbdr[k + width_] * cn + j];
				}
			}
			break;
		}
	}

	int state_;
	HeapMgrA<float> mask_w, mask_h;
	float *mask_ptr_w, *mask_ptr_h;
	HeapMgr<int> imask_w; //用于8位-32位sse优化的中间模板
	float sigmaX_ = 0.6, sigmaY_ = 0.6;
	float su_ = 0, sv_ = 0; //归一化参数
	int width_ = 2, height_ = 2, ksize_w, ksize_h;

	//int row_width, row_step;
	//HeapMgrA<ST> srcBuffRow;
	std::vector< ST > srcBuffRow;

	HeapMgr<int> border_idx;
	int border_offset;
	int *pbdr;

	//列滤波的行标识
	std::vector< float* > col_border_idx;
	int col_border_ptr;

	//functors
	RingBuffer<DT> ringbuff;

	//行滤波器
	RowFilter<ST, DT> rFilter;
	//列滤波器
	ColFilter<DT, DT> cFilter;

	ST *srcbuff = NULL, *src = NULL;
	DT *dst = NULL;
};




}
}

#endif //_HS_FEATURE2D_STD_SIFT_FILTER_HPP_