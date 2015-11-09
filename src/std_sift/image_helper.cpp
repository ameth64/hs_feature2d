#include <algorithm>

#include "hs_feature2d/std_sift/image_helper.hpp"
#include "hs_feature2d/std_sift/ArrayHelper.h"

namespace hs{
namespace feature2d{


int ImageHelper::Rgb2Gray(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
{
	if (img_input.channel() != 3)
	{
		// nothing happens if the channel !=3
		return -1;
	}
	int w = img_input.width(), h = img_input.height();
	int res = img_output.CreateImage(h, w, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE, true);
	if (res != hs::imgio::whole::ImageData::IMAGE_DATA_NO_ERROR)
	{
		return -1;
	}
	//遍历所有像素, 根据以下公式转换为灰度(参考MATLAB的rgb2gray函数)
	// 0.2989 * R + 0.5870 * G + 0.1140 * B
	hs::imgio::whole::ImageData::Byte R, G, B, tmp;
	hs::imgio::whole::ImageData::PByte _outbuff = img_output.GetBuffer();
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			R = img_input.GetByte(i, j, 0);
			G = img_input.GetByte(i, j, 1);
			B = img_input.GetByte(i, j, 2);
			tmp = hs::imgio::whole::ImageData::Byte(0.2989 * R + 0.5870 * G + 0.1140 * B);
			_outbuff[i * w + j] = tmp;
		}
	}

	return int(OprError::OPR_OK);
}

int ImageHelper::GaussianBlur(double sigma, int imw, int imh, 
	hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
{
	int res = 0;
	if (img_input.channel() != 1)
	{
		return res = int(INVALID_CHANNEL);
	}
	int w = img_input.width(), h = img_input.height();
	hs::imgio::whole::ImageData img_tmp;
	res = img_output.CreateImage(w, h, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
	res += img_tmp.CreateImage(w, h, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
	if (res != hs::imgio::whole::ImageData::IMAGE_DATA_NO_ERROR)
	{
		return res = int(CREATE_IMG_FAIL);
	}
	// 使用分离形式做高斯卷积计算
	int mw = (imw > 0) ? (imw) : std::ceil(3 * sigma);
	int mh = (imh > 0) ? (imh) : std::ceil(3 * sigma);
	hs::feature2d::GaussianFilter gsm(sigma, mw, mh);
	GaussianFilter::T_MaskVector& gmsk_w = gsm.GetMask(0);
	GaussianFilter::T_MaskVector& gmsk_h = gsm.GetMask(1);
	float* gmsk_ptr = gmsk_w.GetPtr();
	//pass 1
	hs::imgio::whole::ImageData::PByte _outbuff = img_tmp.GetBuffer(), _inBuff = img_input.GetBuffer();
	int ci = 0, cj = 0, rows = 0;
	double tmp = 0;
	for (int i = 0; i < h; i++)
	{
		rows = i * w;
		for (int j = 0; j < w; j++)
		{
			tmp = 0;
			for (int v = -mw; v <= mw; v++)
			{
				cj = 
				cj = j + v;
				if (cj < 0) cj = -cj;
				if (cj >= w) cj = 2 * w - 2 - cj;
				tmp += _inBuff[rows + cj] * gmsk_ptr[v + mw];
			}
			_outbuff[rows + j] = tmp;
		}
	}

	//pass 2
	gmsk_ptr = gmsk_h.GetPtr();
	_outbuff = img_output.GetBuffer(), _inBuff = img_tmp.GetBuffer();
	for (int i = 0; i < h; i++)
	{
		rows = i * w;
		for (int j = 0; j < w; j++)
		{
			tmp = 0;
			for (int v = -mh; v <= mh; v++)
			{
				cj = i + v;
				if (cj < 0) cj = -cj;
				if (cj >= h) cj = 2 * (h-1) - cj;
				tmp += _inBuff[cj * w + j] * gmsk_ptr[v + mh];
			}
			_outbuff[i * w + j] = int(tmp);
		}
	}
	return res = int(OprError::OPR_OK);
}

int ImageHelper::GaussianBlurAccl(double sigma, int imw, int imh, 
	hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
{
	int res = 0;
	if (img_input.channel() != 1)
	{
		return res = int(INVALID_CHANNEL);
	}
	int w = img_input.width(), h = img_input.height();
	hs::imgio::whole::ImageData img_tmp;
	res = img_output.CreateImage(w, h, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
	res += img_tmp.CreateImage(w, h, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
	if (res != hs::imgio::whole::ImageData::IMAGE_DATA_NO_ERROR)
	{
		return res = int(CREATE_IMG_FAIL);
	}
	// 为满足行计算而声明的临时变量
	hs::feature2d::HeapMgr<double> tmpRow0(w), tmpRow1(w);

	// 使用分离形式做高斯卷积计算
	int mw = (imw > 0) ? (imw) : std::ceil(3 * sigma);
	int mh = (imh > 0) ? (imh) : std::ceil(3 * sigma);
	hs::feature2d::GaussianFilter gsm(sigma, mw, mh);
	GaussianFilter::T_MaskVector& gmsk_w = gsm.GetMask(0);
	GaussianFilter::T_MaskVector& gmsk_h = gsm.GetMask(1);
	float* gmsk_ptr = gmsk_w.GetPtr();
	//pass 1
	hs::imgio::whole::ImageData::PByte _outbuff = img_tmp.GetBuffer(), 
		_inBuff = img_input.GetBuffer(), _resbuff = img_output.GetBuffer();
	int ci = 0, cj = 0, ck = 0, cl = mw + w - 1, rows = 0, v;
	double tmp = 0;
	for (int i = 0; i < h; i++)
	{
		rows = i * w;
		for (int j = 0; j < w; j++)
		{
			tmp = 0; ck = j - mw;
			for (v = 2*mw, cj=2*mw+ck; v >=0; v--,cj--)
			{
				ci = (cj < 0) ? (abs(cj)) : ( cj>=w ? (cl-v) : cj );
				tmp += _inBuff[rows + ci] * gmsk_ptr[v];
			}
			_outbuff[rows + j] = tmp;
		}
	}

	//pass2
	double* dptr = tmpRow0.GetPtr();
	cl = mw + h - 1;
	for (int j = 0; j < h; j++)
	{
		rows = j * w; ck = j - mw;
		for (v = 2 * mw, cj = 2 * mw + ck; v >= 0; v--, cj--)
		{
			ci = w * ( (cj < 0) ? (abs(cj)) : (cj>h ? (cl - v) : cj) );
			for (int u = 0; u < w; u++)
			{
				dptr[u] += gmsk_ptr[v] * _outbuff[ci+u];
			}
		}
		for (int u = 0; u < w; u++)
		{
			_resbuff[rows + u] = int(dptr[u]);
			dptr[u] = 0.0;
		}
	}
	return res = int(OprError::OPR_OK);
}

int ImageHelper::GaussianBlurSSE(double sigma, int imw, int imh, 
	hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
{
	int cn = img_input.channel(), res = 0;
	if (cn != 1)
	{
		return res = int(INVALID_CHANNEL);
	}
	int w = img_input.width(), h = img_input.height();
	
	// 创建输出图像
	res = img_output.CreateImage(w, h, 1, 8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
	// 创建中间缓存图像
	hs::imgio::whole::ImageData img_tmp;
	res *= img_tmp.CreateImage(w, h, 4, 8, hs::imgio::whole::ImageData::IMAGE_ARGB);
	if (res != hs::imgio::whole::ImageData::IMAGE_DATA_NO_ERROR)
	{
		return res = int(CREATE_IMG_FAIL);
	}

	// 使用分离形式做高斯卷积计算
	int mw = (imw > 0) ? (imw) : std::ceil(3 * sigma); //模板宽
	int mh = (imh > 0) ? (imh) : std::ceil(3 * sigma); //模板高
	hs::feature2d::GaussianFilter gsm(sigma, mw, mh);
	GaussianFilter::T_MaskVector& gmsk_w = gsm.GetMask(0);
	GaussianFilter::T_MaskVector& gmsk_h = gsm.GetMask(1);
	// sse优化
	// 指定对齐分配内存
	std::size_t src_buff_size = (w + gmsk_w.GetSize() - 1) * cn,
		tmp_buff_size = alignSize((w + gmsk_w.GetSize() - 1) * cn * 4, F2D_PTR_ALIGN); //中间数组使用32位,需要尺寸对齐

	// 将模板值映射至0-255区间
	hs::feature2d::HeapMgrA<int> kernel_x(gmsk_w.GetSize()), kernel_y(gmsk_h.GetSize());
	int *_kx = kernel_x.GetPtr(), *_ky = kernel_y.GetPtr();
	float* msk_ptr = gmsk_w.GetPtr();
	for (int k = mw - 1; k >= 0; k--)
	{
		_kx[k] = int(msk_ptr[k] * (1 << 8));
	}
	msk_ptr = gmsk_h.GetPtr();
	for (int k = mh - 1; k >= 0; k--)
	{
		_ky[k] = int(msk_ptr[k] * (1 << 8));
	}

	// 行变换
	int width = w * cn;
	int i = 0, k = 0;
	hs::feature2d::HeapMgrA<hs::imgio::whole::ImageData::Byte> tmpBuff(tmp_buff_size);
	for (int y = 0; y < h; y++)
	{
		const hs::imgio::whole::ImageData::PByte src = img_input.GetLine(y);
		sseRowFilter(src, tmpBuff.GetPtr(), w, cn, _kx, mw);
	}
	

	return res = int(OprError::OPR_OK);
}

int ImageHelper::sseRowFilter(const hs::imgio::whole::ImageData::PByte _src, hs::imgio::whole::ImageData::PByte dst, int w, int cn, int* _kx, int mw)
{
	int width = w * cn;
	int i = 0, k = 0;
	hs::imgio::whole::ImageData::PByte src;
	for (; i < width - 16; i += 16)
	{
		src = _src + i;
		__m128i f, z = _mm_setzero_si128(), s0 = z, s1 = z, s2 = z, s3 = z;
		__m128i x0, x1, x2, x3;
		//遍历模板
		for (; k < mw; k++, src += cn)
		{
			f = _mm_cvtsi32_si128(_kx[k]);
			f = _mm_shuffle_epi32(f, 0);
			f = _mm_packs_epi32(f, f);

			x0 = _mm_loadu_si128((const __m128i*)src);
			x2 = _mm_unpackhi_epi8(x0, z);
			x0 = _mm_unpacklo_epi8(x0, z);
			x1 = _mm_mulhi_epi16(x0, f);
			x3 = _mm_mulhi_epi16(x2, f);
			x0 = _mm_mullo_epi16(x0, f);
			x2 = _mm_mullo_epi16(x2, f);

			s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
			s1 = _mm_add_epi32(s1, _mm_unpackhi_epi16(x0, x1));
			s2 = _mm_add_epi32(s2, _mm_unpacklo_epi16(x2, x3));
			s3 = _mm_add_epi32(s3, _mm_unpackhi_epi16(x2, x3));
		}
		_mm_store_si128((__m128i*)(dst + i), s0);
		_mm_store_si128((__m128i*)(dst + i + 4), s1);
		_mm_store_si128((__m128i*)(dst + i + 8), s2);
		_mm_store_si128((__m128i*)(dst + i + 12), s3);
	}

	//不满足16字节对齐的剩余元素, 以4字节为步长处理
	for (; i <= width - 4; i += 4)
	{
		src = _src + i;
		__m128i f, z = _mm_setzero_si128(), s0 = z, x0, x1;

		for (k = 0; k < mw; k++, src += cn)
		{
			f = _mm_cvtsi32_si128(_kx[k]);
			f = _mm_shuffle_epi32(f, 0);
			f = _mm_packs_epi32(f, f);

			x0 = _mm_cvtsi32_si128(*(const int*)src);
			x0 = _mm_unpacklo_epi8(x0, z);
			x1 = _mm_mulhi_epi16(x0, f);
			x0 = _mm_mullo_epi16(x0, f);
			s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
		}
		_mm_store_si128((__m128i*)(dst + i), s0);
	}

	//小于步长4字节的逐个处理
	for (; i < width; i++)
	{
		src = _src + i;
		int s0 = _kx[0] * src[0];
		for (k = 1; k < mw; k++)
		{
			src += cn;
			s0 += _kx[k] * src[0];
		}
		dst[i] = s0;
	}
	return 0;
}






}
}
