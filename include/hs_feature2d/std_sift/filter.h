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
#include "hs_feature2d/std_sift/base_type.h"
//#include "hs_feature2d/std_sift/numeric_solver.hpp"
#include "hs_feature2d/std_sift/ArrayHelper.h"


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
	inline T* Line(int i){ return linePtr[ i % buffLines ]; }
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
	void operator()(ST* srcbuff, DT* dst, float* kernal, int& kw, int& w, int& cn)
	{
		int row_size = w * cn, j, k;
		//初始化目标行元素
		memset(dst, 0, sizeof(DT) * row_size);
		//遍历行模板
		for (k = 0; k < kw * 2 + 1; k++, srcbuff += cn)
		{
			//遍历行元素
			for (j = 0; j < row_size; j++)
			{
				dst[j] += DT(srcbuff[j] * kernal[k]);
			}
		}
	}
};

//使用sse优化的行滤波functor
template<> struct RowFilter<float, float>
{
	forceinline void operator()(float* srcbuff, float* dst, float* mask_ptr, int& kw, int& w, int& cn)
	{
#ifdef  _USE_SSE2_
		int row_width = w*cn, i, j, k, _ksize = kw * 2 + 1;
		float* src;
		__m128 f, s0, s1, x0, x1;
		for (j = 0; j <= row_width - 8; j += 8)
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
		for (i = 0; i < row_size; i++)
		{
			src = srcbuff + i;
			dst[i] *= 0;
			for (k = 0; k < _ksize; k++, src += cn)
			{
				dst[i] += (*src) * kernal[k];
			}
		}
#endif // _USE_SSE2_
	}
};

//通用的列滤波functor
template<typename ST, typename DT>
struct ColFilter
{
	inline void operator()(RingBuffer<float>& ringbuff, float* dst, float* mask_ptr_h, int& idx, int& height_, int& w, int& h, int& cn)
	{
		int row_width = w * cn, j, k, c;
		float *ptmp;
		memset(dst, 0, row_width * sizeof(DT));
		for (k = 0; k < height_ * 2 + 1; k++)
		{
			c = idx + k - height_ * 2;
			c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			ptmp = ringbuff.Line(c);
			for (j = 0; j < row_width; j++)
			{
				dst[j] += mask_ptr_h[k] * ptmp[j];
			}
		}
	}
};

//针对float类型的加速版本
template<> struct ColFilter<float, float>
{
	forceinline void operator()(RingBuffer<float>& ringbuff, float* dst, float* mask_ptr_h, int& idx, int& height_, int& w, int& h, int& cn)
	{
#ifdef  _USE_SSE2_
		int c, j, k, row_width = w * cn;
		float *ptmp;
		__m128 f, s0, s1, s2, s3, x0, x1;
		memset(dst, 0, row_width * sizeof(float));
		for (k = 0; k < height_ * 2 + 1; k++)
		{
			c = idx + k - height_ * 2;
			c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			ptmp = ringbuff.Line(c);
			f = _mm_load_ps(mask_ptr_h + k);
			f = _mm_shuffle_ps(f, f, 0);
			for (j = 0; j <= row_width - 8; j += 8)
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
			for (; j <= row_width - 4; j += 4)
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
#endif
	}
};


//高斯滤波类
class HS_EXPORT GaussianFilter
{
public:
	typedef HeapMgr<float> T_MaskVector;
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

	//主要操作入口
	template<typename ST, typename DT = float>
	int Apply(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
	{
		int w = img_input.width(), h = img_input.height(), cn = img_input.channel();
		int row_width = w * cn, row_step = w * cn * sizeof(DT);
		int res = 0;
		res = img_output.CreateImage(w, h, cn, sizeof(DT) * 8, img_input.color_type(), true);
		
		int i, j, k;
		float *mask_ptr, *mask_ptr_h;

		//镜像边界的元素索引
		HeapMgr<int> border_idx(width_ * 2), border_idx_h(height_ * 2);
		int *pbdr = border_idx.GetPtr();
		int border_offset = width_*cn;
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = (width_ - i)*cn, pbdr[i + width_] = (w - 2 - i)*cn;
		}
		pbdr = border_idx_h.GetPtr();
		for (i = 0; i < height_; i++)
		{
			pbdr[i] = (height_ - i), pbdr[i + height_] = (h - 2 - i);
		}

		// 行滤波缓冲
		RingBuffer<DT> ringbuff(w*cn, std::min(2*height_ + 1, h));
		//行滤波器
		RowFilter<ST, DT> filter;
		//列滤波器
		ColFilter<ST, DT> cFilter;
		
		//源与目标图像缓冲行
		HeapMgrA<ST> srcBuffRow(row_width + width_ * 2 * cn);
		HeapMgrA<DT> dstBuffRow(row_width);
		ST *srcbuff, *src;
		DT *dst = NULL, *ptmp = NULL, *src0, *tgt;
		size_t minAlignedRows = row_width % (HS_BYTE_ALIGN / sizeof(DT)); //可满足字节对齐的最小行数
		int c = 0, startH = 0, _ksize = 2 * width_ + 1;
		mask_ptr = mask_w.GetPtr(), mask_ptr_h = mask_h.GetPtr();;
		for (i = 0; i < h; i++)
		{
			srcbuff = srcBuffRow.GetPtr();
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_width * sizeof(ST));
			for (k = 0; k < width_; k++)
			{
				for (j = 0; j < cn; j++)
				{
					srcbuff[k*cn + j] = src[pbdr[k] * cn + j], srcbuff[(border_offset + w + k)*cn + j] = src[pbdr[k + width_] * cn + j];
				}
			}
			//设置中间缓冲行指针
			if (i >= ringbuff.BuffLines()){
				ringbuff.Shift();
				dst = ringbuff.LastLine();
			}
			else
			{
				dst = ringbuff.Line(i);
			}
#ifdef _USE_SSE2_

			//封装调用
			filter(srcbuff, dst, mask_ptr, width_, w, cn);

			////直接展开的代码段
			//__m128 f, s0, s1, x0, x1;
			//for (j = 0; j <= row_width - 8; j += 8)
			//{
			//	src = srcbuff + j;
			//	s0 = s1 = _mm_setzero_ps();
			//	for (k = 0; k < _ksize; k++, src += cn)
			//	{
			//		//将单个32f的模板元素阵列至128位
			//		f = _mm_load_ss(mask_ptr + k);
			//		f = _mm_shuffle_ps(f, f, 0);

			//		x0 = _mm_loadu_ps(src);
			//		x1 = _mm_loadu_ps(src + 4);
			//		s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
			//		s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
			//	}
			//	_mm_store_ps(dst + j, s0);
			//	_mm_store_ps(dst + j + 4, s1);
			//}

			//for (; j <= row_width - 4; j += 4)
			//{
			//	src = srcbuff + j;
			//	__m128 f, s0 = _mm_setzero_ps(), x0;
			//	for (k = 0; k < _ksize; k++, src += cn)
			//	{
			//		f = _mm_load_ss(mask_ptr + k);
			//		f = _mm_shuffle_ps(f, f, 0);

			//		x0 = _mm_loadu_ps(src);
			//		s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
			//	}
			//	_mm_store_ps(dst + j, s0);
			//}

			//for (; j < row_width; j++)
			//{
			//	src = srcbuff + j;
			//	dst[j] *= 0;
			//	for (k = 0; k < _ksize; k++, src += cn)
			//	{
			//		dst[j] += (*src) * mask_ptr[k];
			//	}
			//}
			////直接展开的代码段

#endif // _USE_SSE2_



			//测试, 直接将行滤波结果写入
			//tgt = img_output.GetLineT<DT>(i);
			//memcpy(tgt, dst, row_width*sizeof(DT));

			if (i >= height_) //可同步进行列滤波
			{ 
				startH = i - height_;
#ifdef _USE_SSE2_
				dst = img_output.GetLineT<DT>(startH);

				//封装调用ColFilter
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);

				////直接展开的代码段
				//__m128 f, s0, s1, s2, s3, x0, x1;
				//for (k = 0; k < height_*2+1; k++)
				//{
				//	c = i + k - height_ * 2;
				//	c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
				//	ptmp = ringbuff.Line(c);
				//	f = _mm_load_ps(mask_ptr_h + k);
				//	f = _mm_shuffle_ps(f, f, 0);
				//	for (j = 0; j <= row_width - 8; j += 8)
				//	{

				//		x0 = _mm_load_ps(ptmp + j);
				//		x1 = _mm_load_ps(ptmp + j + 4);
				//		s0 = _mm_loadu_ps(dst + j);
				//		s1 = _mm_loadu_ps(dst + j + 4);
				//		s2 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				//		s3 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
				//		_mm_storeu_ps(dst + j, s2);
				//		_mm_storeu_ps(dst + j + 4, s3);
				//	}
				//	for (; j <= row_width - 4; j+=4)
				//	{

				//		x0 = _mm_load_ps(ptmp + j);
				//		s0 = _mm_loadu_ps(dst + j);
				//		s1 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
				//		_mm_storeu_ps(dst + j, s1);
				//	}
				//	for (; j < row_width; j++)
				//	{
				//		if (k == 0)
				//			dst[j] *= 0;
				//		dst[j] += mask_ptr_h[k] * ptmp[j];
				//	}
				//}
				////直接展开的代码段

				//tgt = img_output.GetLineT<DT>(startH);
				//memcpy((void*)tgt, (void*)dst, row_width*sizeof(DT));
#else
				dst = img_output.GetLineT<DT>(startH);
				memset((void*)dst, 0, row_width * sizeof(DT));
				//遍历模板
				for (k = 0; k < height_*2+1; k++)
				{
					c = i + k - height_ * 2;
					c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
					ptmp = ringbuff.Line(c);
					//ptmp = ringbuff.OffsetLine(0);
					for (j = 0; j < row_width; j++)
					{
						dst[j] += mask_ptr_h[k] * ptmp[j];
						//dst[j] = ptmp[j];
					}
				}
#endif // _USE_SSE2_
			}
		}

		if (startH < h - 1)
		{
#ifdef _USE_SSE2_
			//封装调用
			for (i = startH; i < h; i++)
			{
				dst = img_output.GetLineT<DT>(i);
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}

			////直接展开的代码段
			//mask_ptr_h = mask_h.GetPtr();
			//__m128 f, s0, s1, s2, s3, x0, x1;
			//for (i = startH; i < h; i++)
			//{
			//	dst = img_output.GetLineT<DT>(startH);
			//	memset(dst, 0, row_width * sizeof(DT));
			//	for (k = 0; k < height_*2+1; k++)
			//	{
			//		c = i + k - height_ * 2;
			//		c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
			//		ptmp = ringbuff.Line(c);
			//		//const float* src = srcbuff + i;
			//		
			//		f = _mm_load_ps(mask_ptr_h + k);
			//		f = _mm_shuffle_ps(f, f, 0);
			//		for (j = 0; j <= row_width - 8; j += 8)
			//		{

			//			x0 = _mm_load_ps(ptmp + j);
			//			x1 = _mm_load_ps(ptmp + j + 4);
			//			s0 = _mm_loadu_ps(dst + j);
			//			s1 = _mm_loadu_ps(dst + j + 4);
			//			s2 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
			//			s3 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
			//			_mm_storeu_ps(dst + j, s2);
			//			_mm_storeu_ps(dst + j + 4, s3);
			//		}
			//		for (; j <= row_width - 4; j += 4)
			//		{

			//			x0 = _mm_load_ps(ptmp + j);
			//			s0 = _mm_loadu_ps(dst + j);
			//			s1 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
			//			_mm_storeu_ps(dst + j, s1);
			//		}
			//		for (; j < row_width; j++)
			//		{
			//			if (k == 0)
			//				dst[j] *= 0;
			//			dst[j] += mask_ptr_h[k] * ptmp[j];
			//		}
			//	}
			//}
			////直接展开的代码段

#else
			mask_ptr_h = mask_h.GetPtr();
			for (; startH < h; startH++)
			{
				dst = img_output.GetLineT<DT>(startH);
				memset(dst, 0, row_width * sizeof(DT));
				for (k = 0; k < height_*2+1; k++)
				{
					c = startH + k - height_ * 2;
					c = c < 0 ? -c : (c >= h ? 2 * h - c - 2 : c);
					ptmp = ringbuff.OffsetLine(c);
					//ptmp = ringbuff.OffsetLine(0);
					for (j = 0; j < row_width; j++)
					{
						dst[j] += mask_ptr_h[k] * ptmp[j];
					}
				}
			}
#endif // _USE_SSE2_
		}
		return res;
	};


private:
	void initMask();

	template<typename ST, typename DT>
	int rowFilter(const hs::imgio::whole::ImageData& img_src, hs::imgio::whole::ImageData& img_dst)
	{
		RowFilter<ST, DT> filter;
		int w = img_src.width(), h = img_src.height(), cn = img_src.channel(), depth = img_src.bit_depth();
		//创建目标图像, 注意采取16字节对齐
#ifdef _USE_SSE2_
		int res = img_dst.CreateImage(w, h, cn, sizeof(DT)*Mat::ByteDepth, img_src.color_type(), false);
#else
		int res = img_dst.CreateImage(w, h, cn, sizeof(DT)*Mat::ByteDepth, img_src.color_type());
#endif // _USE_SSE2_

		
		DT* pdst = (DT*)img_dst.GetBuffer();
		
		int i, j, k;
		//缓冲行
		hs::feature2d::HeapMgr<ST> srcBuffRow((w + width_ * 2) * cn);
		//镜像边界的元素索引
		HeapMgr<int> border_idx(width_ * 2);
		int *pbdr = border_idx.GetPtr();
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = (width_ - i)*cn, pbdr[i + width_] = (w - 2 - i)*cn;
		}
		//
		int row_size = w * cn, buff_size = (w + width_ * 2) * cn;
		ST* srcbuff = srcBuffRow.GetPtr();
		ST* src = NULL;
		DT* dst = NULL;
		float tmp = 0, *mask_ptr = mask_w.GetPtr();
		for (i = 0; i < h; i++)
		{
			src = (ST*)img_src.GetLine(i);
			dst = (DT*)img_dst.GetLine(i);
			srcbuff = srcBuffRow.GetPtr();
			memcpy(srcbuff + width_*cn, src, w * cn * sizeof(ST));
			//设置镜像边界元素
			for (k = 0; k < width_; k++)
			{
				for (j = 0; j < cn; j++)
				{
					srcbuff[k*cn + j] = src[pbdr[k] * cn + j], srcbuff[(w + k)*cn + j] = src[pbdr[k + width_] * cn + j];
				}
			}
			filter(srcbuff, dst, mask_ptr, width_, w, cn);

		}
		return res;
	};

	template<typename ST, typename DT>
	int columnFilter(const hs::imgio::whole::ImageData& img_src, hs::imgio::whole::ImageData& img_dst)
	{
		int w = img_src.width(), h = img_src.height(), cn = img_src.channel(), depth = img_src.bit_depth();
		int res = img_dst.CreateImage(w, h, cn, depth, img_src.color_type());
		DT* pdst = img_dst.GetBufferT<DT>();
		memset(pdst, 0, sizeof(DT) * w * h * cn);
		//
		int row_size = w * cn, col_size = (h + height_ * 2);
		int i, j, k;

		//缓冲行
		hs::feature2d::HeapMgr<ST> srcRowPtr(col_size);
		//镜像边界的元素索引
		HeapMgr<int> border_idx(height_ * 2);
		int *pbdr = border_idx.GetPtr();
		for (i = 0; i < height_; i++)
		{
			pbdr[i] = (height_ - i), pbdr[i + height_] = (h - 2 - i);
		}

		float tmp = 0, *mask_ptr = mask_h.GetPtr();
		int row_index = 0;
		ST* src = NULL;
		DT* dst = NULL;
		for (i = 0; i < h; i++)
		{
			dst = (DT*)img_dst.GetLine(i);
			//调用ColFilter的functor
			for (k = 0; k < height_ * 2 + 1; k++)
			{
				row_index = i + k - height_;
				src = (row_index < 0) ? (ST*)(img_src.GetLine(pbdr[row_index + height_])) 
					: (row_index >= h ? (ST*)(img_src.GetLine(pbdr[row_index - h + height_])) : (ST*)(img_src.GetLine(row_index)));
				for (j = 0; j < row_size; j++)
				{
					dst[j] += src[j] * mask_ptr[k];
				}
			}
		}
		return res;
	};

private:
	int state_;
	HeapMgr<float> mask_w, mask_h;
	HeapMgr<int> imask_w; //用于sse优化的中间模板
	float sigmaX_ = 0.6, sigmaY_ = 0.6;
	float su_ = 0, sv_ = 0; //归一化参数
	int width_ = 2, height_ = 2;
};




}
}

#endif //_HS_FEATURE2D_STD_SIFT_FILTER_HPP_