#ifndef _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_
#define _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_

#include <string>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <emmintrin.h>
#include <omp.h>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_image_io/whole_io/image_io.hpp"
#include "hs_feature2d/std_sift/base_type.hpp"
#include "hs_feature2d/std_sift/matrix.hpp"
#include "hs_feature2d/std_sift/ArrayHelper.hpp"
#include "hs_feature2d/std_sift/filter.hpp"


//
namespace hs
{
namespace feature2d
{

typedef hs::imgio::whole::ImageData Image;

class HS_EXPORT ImageHelper: public hs::imgio::whole::ImageIO
{
public:
	//操作结果标识
	enum OprError
	{
		CREATE_IMG_FAIL = -1,
		OPR_OK = 0,
		INVALID_CHANNEL,
		INVALID_SUBSTRACT_IMG
	};

	//图像缩放标识
	enum
	{
		INVALID_RESIZE = -1,
		INTER_NEAREST = 0,	//!< nearest neighbor interpolation
		INTER_LINEAR = 1,	//!< bilinear interpolation
		INTER_CUBIC = 2,	//!< bicubic interpolation
		INTER_AREA = 3		//!< area-based (or super) interpolation
	};
	
	ImageHelper(){};
	~ImageHelper(){};

	// RGB与灰度图转换
	template<typename ST, typename DT>
	static inline int rowRgb2Gray(ST* src, DT* dst, int row_width)
	{
		int i = 0;
		for (; i < row_width; i++)
		{
			dst[i] = saturate_cast<DT>(src[i * 3] * 0.2989 + src[i * 3 + 1] * 0.5870 + src[i * 3 + 2] * 0.1140);
		}
	}

	template<typename ST, typename DT>
	static int Rgb2Gray(hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output)
	{
		if (img_input.channel() != 3)
		{
			// nothing happens if the channel !=3
			return -1;
		}
		int w = img_input.width(), h = img_input.height();
		int res = img_output.CreateImage(h, w, 1, sizeof(DT)*8, hs::imgio::whole::ImageData::IMAGE_GRAYSCALE);
		if (res != hs::imgio::whole::ImageData::IMAGE_DATA_NO_ERROR)
		{
			return -1;
		}
		//根据以下公式转换为灰度(参考MATLAB的rgb2gray函数)
		// 0.2989 * R + 0.5870 * G + 0.1140 * B
		DT* _outbuff = img_output.GetBufferT<DT>();
		ST* _inbuff = img_input.GetBufferT<ST>();

		int len = w * h, i = 0;
#ifdef _OPENMP
#pragma omp parallel for
		for (i = 0; i < len; i++)
		{
			//if (i == 0)
			//	std::cout << "omp_get_num_threads = " << omp_get_num_threads() << std::endl;
			_outbuff[i] = saturate_cast<DT>(_inbuff[i * 3] * 0.2989 + _inbuff[i * 3 + 1] * 0.5870 + _inbuff[i * 3 + 2] * 0.1140);
		}
#else
		for (i = 0; i < len; i++)
		{
			_outbuff[i] = saturate_cast<DT>(_inbuff[i * 3] * 0.2989 + _inbuff[i * 3 + 1] * 0.5870 + _inbuff[i * 3 + 2] * 0.1140);
		}
		//for (int i = 0; i < h; i++)
		//{
		//	for (int j = 0; j < w; j++)
		//	{
		//		R = img_input.GetByte(i, j, 0);
		//		G = img_input.GetByte(i, j, 1);
		//		B = img_input.GetByte(i, j, 2);
		//		tmp = hs::imgio::whole::ImageData::Byte(0.2989 * R + 0.5870 * G + 0.1140 * B);
		//		_outbuff[i * w + j] = tmp;
		//	}
		//}
#endif // _OPENMP
		return int(OprError::OPR_OK);
	};

	// 图像缩放
	template<typename T = hs::imgio::whole::ImageData::Byte>
	static int Resize(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output, size_t dstw, size_t dsth, int method)
	{
		int res = 0;
		if (dstw * dsth <= 0)
		{
			return int(INVALID_RESIZE);
		}
		int srcw = img_input.width(), srch = img_input.height();
		float ratiox = float(srcw) / dstw, ratioy = float(srch) / dsth;
		img_output.CreateImage(dstw, dsth, img_input.channel(), img_input.bit_depth(), img_input.color_type());
		int i, j, k, cn = img_input.channel();
		int src_step = srcw*cn, dst_step = dstw * cn;
		T* psrc = (T*)img_input.GetBuffer();
		T* pdst = (T*)img_output.GetBuffer();
		if (method == INTER_NEAREST)
		{
			int srcx, srcy;
			for (i = 0; i < dsth; i++)
			{
				srcy = std::round(i * ratioy);
				for (j = 0; j < dstw; j++)
				{
					srcx = std::round(j * ratiox);
					for (k = 0; k < cn; k++)
					{
						pdst[i * dst_step + j * cn + k] = psrc[srcy * src_step + srcx * cn + k];
					}
				}
			}
		}
		if (method == INTER_LINEAR)
		{
			float srcx, srcy, dx, dy;
			float tmpX0, tmpX1, tmpY0, tmpY1, tmpDstX, tmpDstY;
			int corners[4];
			for (i = 0; i < dsth; i++)
			{
				srcy = i * ratioy, corners[2] = std::floor(srcy), corners[3] = std::ceil(srcy), dy = srcy - corners[2];
				if (corners[3] >= srch){
					corners[3] = srch - 1; dy *= 0;
				}
					
				for (j = 0; j < dstw; j++)
				{
					srcx = j * ratiox, dx = srcx - std::floor(srcx);
					for (k = 0; k < cn; k++)
					{
						corners[0] = std::floor(srcx), corners[1] = std::ceil(srcx);
						if (corners[1] >= srcw){
							corners[1] = srcw - 1; dx *= 0;
						}
						if (dx > FLT_EPSILON)
						{
							tmpX0 = psrc[corners[2] * src_step + corners[0] * cn + k];
							tmpX1 = psrc[corners[2] * src_step + corners[1] * cn + k];

							tmpDstX = tmpX0 * (corners[1] - srcx) + tmpX1 * (srcx - corners[0]);
						}
						else
						{
							tmpDstX = psrc[corners[2] * src_step + corners[0] * cn + k];
						}
						if (dy > FLT_EPSILON)
						{
							if (dx > FLT_EPSILON)
							{
								tmpY0 = psrc[corners[3] * src_step + corners[0] * cn + k];
								tmpY1 = psrc[corners[3] * src_step + corners[1] * cn + k];
								tmpDstY = tmpY0 * (corners[1] - srcx) + tmpY1 * (srcx - corners[0]);
							}
							else
							{
								tmpDstY = psrc[corners[3] * src_step + corners[0] * cn + k];
							}

							pdst[i * dst_step + j * cn + k] = tmpDstX * (corners[3] - srcy) + tmpDstY * (srcy - corners[2]);
						}
						else
						{
							pdst[i * dst_step + j * cn + k] = tmpDstX;
						}
					}
				}
			}
		}

		return res;
	};

	//数据类型转换
private:
	template<typename ST, typename DT> static inline void rowConvert(ST* src, DT* dst, int row_width)
	{
		int j = 0;
		for (; j <= row_width - 4; j += 4)
		{
			dst[j] = saturate_cast<DT>(src[j]);
			dst[j + 1] = saturate_cast<DT>(src[j + 1]);
			dst[j + 2] = saturate_cast<DT>(src[j + 2]);
			dst[j + 3] = saturate_cast<DT>(src[j + 3]);
		}
		for (; j < row_width; j++)
		{
			dst[j] = saturate_cast<DT>(src[j]);
		}
	}

public:
	template<typename ST, typename DT>
	static void ConvertDataType(const hs::imgio::whole::ImageData& img_src, hs::imgio::whole::ImageData& img_dst)
	{
		size_t w = img_src.width(), h = img_src.height(), cn = img_src.channel();
		img_dst.CreateImage(w, h, cn, sizeof(DT) * 8, img_src.color_type());
		ST* src = (ST*)img_src.GetBuffer();
		DT* dst = img_dst.GetBufferT<DT>();
		
		int row_width = cn * w, i;

#ifdef _OPENMP
#pragma omp parallel for
		for (i = 0; i < h; i++)
		{
			rowConvert<ST, DT>(src + i*row_width, dst + i*row_width, row_width);
		}
#else
		for (i = 0; i < h; i++, src += row_width, dst += row_width)
		{
			for (int j = 0; j <= row_width - 4; j += 4)
			{
				dst[j] = saturate_cast<DT>(src[j]);
				dst[j + 1] = saturate_cast<DT>(src[j + 1]);
				dst[j + 2] = saturate_cast<DT>(src[j + 2]);
				dst[j + 3] = saturate_cast<DT>(src[j + 3]);
			}
			for (; j < row_width; j++)
			{
				dst[j] = saturate_cast<DT>(src[j]);
			}
		}

#endif // _OPENMP
	};

	//差分运算, 用于生成DOG高斯差分金字塔
	template<typename T>
	static int Subtract(const Image& src1, const Image& src2, Image& dst)
	{
		int res = 0;
		T* psrc1 = (T*)src1.GetBuffer();
		T* psrc2 = (T*)src2.GetBuffer();
		int w1 = src1.width(), h1 = src1.height(), cn1 = src1.channel(), 
			w2 = src2.width(), h2 = src2.height(), cn2 = src2.channel();
		if(!(w1 == w2 && h1 == h2 && cn1 == cn2))
			return res = int(INVALID_SUBSTRACT_IMG);
		res = dst.CreateImage(w1, h1, cn1, sizeof(T)*Mat::ByteDepth, src1.color_type());
		T* pdst = dst.GetBufferT<T>();
		for (int i = 0, len = w1*h1*cn1; i < len; i++)
		{
			pdst[i] = psrc1[i] - psrc2[i];
		}
		return res;
	}


private:


};




}
}


#endif