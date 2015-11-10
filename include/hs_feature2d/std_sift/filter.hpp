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


//�ߴ���뷽��
static inline size_t alignSize(size_t sz, int n)
{
	assert((n & (n - 1)) == 0); // n ������2��������
	return (sz + n - 1) & -n;
}

//ָ�����
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
	return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

//������ʽ��˹�˲����л��������, ��16�ֽڶ��뷽ʽ�����ڴ�
template<typename T>
struct RingBuffer
{
	//ע����ڶ�ͨ��ͼ��, �гߴ�rw=������*ͨ����
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

//ͨ�����˲�functor
template<typename ST, typename DT>
struct RowFilter
{
	void operator()(ST* srcbuff, DT* dst, float* kernal, int& kw, int& w, int& cn)
	{
		int row_size = w * cn, j, k;
		//��ʼ��Ŀ����Ԫ��
		memset(dst, 0, sizeof(DT) * row_size);
		//������ģ��
		for (k = 0; k < kw * 2 + 1; k++, srcbuff += cn)
		{
			//������Ԫ��
			for (j = 0; j < row_size; j++)
			{
				dst[j] += DT(srcbuff[j] * kernal[k]);
			}
		}
	}
};

//���float���͵�SSE���ٰ汾
template<> struct RowFilter<float, float>
{
	forceinline void operator()(float* srcbuff, float* dst, float* mask_ptr, int& kw, int& w, int& cn)
	{
#ifdef  _USE_SSE2_
		int row_width = w*cn, j, k, _ksize = kw * 2 + 1;
		float* src;
		__m128 f, s0, s1, x0, x1;
		for (j = 0; j <= row_width - 8; j += 8)
		{
			src = srcbuff + j;
			s0 = s1 = _mm_setzero_ps();
			for (k = 0; k < _ksize; k++, src += cn)
			{
				//������32f��ģ��Ԫ��������128λ
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
		//��ʼ��Ŀ����Ԫ��
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

//ͨ�õ����˲�functor
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

//���float���͵�SSE���ٰ汾
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
#else
		int c, j, k, row_width = w * cn;
		float *ptmp;
		memset(dst, 0, row_width * sizeof(float));
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
#endif
	}
};


//��˹�˲���
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

	//��Ҫ�������
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

		//����߽��Ԫ������
		HeapMgr<int> border_idx(width_ * 2);
		int *pbdr = border_idx.GetPtr();
		int border_offset = width_*cn;
		for (i = 0; i < width_; i++)
		{
			pbdr[i] = (width_ - i)*cn, pbdr[i + width_] = row_width - (2 + i)*cn;
		}

		// ���˲�����
		RingBuffer<DT> ringbuff(row_width, std::min(2 * height_ + 1, h));
		//���˲���
		RowFilter<ST, DT> rFilter;
		//���˲���
		ColFilter<ST, DT> cFilter;
		
		//Դ��Ŀ��ͼ�񻺳���
		HeapMgrA<ST> srcBuffRow(row_width + width_ * 2 * cn);
		ST *srcbuff, *src;
		DT *dst = NULL, *ptmp = NULL;
		
		mask_ptr = mask_w.GetPtr(), mask_ptr_h = mask_h.GetPtr();;
		for (i = 0; i < h; i++)
		{
			//����Դͼ��Ļ�����
			srcbuff = srcBuffRow.GetPtr();
			src = (ST*)img_input.GetLine(i);
			memcpy((void*)(srcbuff + border_offset), (void*)src, row_width * sizeof(ST));
			//���û����еľ���߽�
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
			
			//�����м价�λ�����ָ��
			if (i >= ringbuff.BuffLines()){
				ringbuff.Shift();
				dst = ringbuff.LastLine();
			}
			else
			{
				dst = ringbuff.Line(i);
			}
			//�������˲�functor
			rFilter(srcbuff, dst, mask_ptr, width_, w, cn);

			if (i >= height_) //�ѻ����㹻����, ��ͬ����ʼ���˲�
			{ 
				startH = i - height_;
				dst = img_output.GetLineT<DT>(startH);
				//�������˲�functor
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}
		}

		if (startH < h - 1)
		{
			//����ʣ�����
			for (i = startH; i < h; i++)
			{
				dst = img_output.GetLineT<DT>(i);
				cFilter(ringbuff, dst, mask_ptr_h, i, height_, w, h, cn);
			}
		}
		return res;
	};


private:
	void initMask();

	int state_;
	HeapMgr<float> mask_w, mask_h;
	HeapMgr<int> imask_w; //����8λ-32λsse�Ż����м�ģ��
	float sigmaX_ = 0.6, sigmaY_ = 0.6;
	float su_ = 0, sv_ = 0; //��һ������
	int width_ = 2, height_ = 2;
};




}
}

#endif //_HS_FEATURE2D_STD_SIFT_FILTER_HPP_