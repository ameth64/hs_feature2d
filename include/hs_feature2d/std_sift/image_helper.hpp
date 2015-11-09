#ifndef _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_
#define _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_

#include <string>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <emmintrin.h>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_image_io/whole_io/image_io.hpp"
#include "hs_feature2d/std_sift/base_type.h"
#include "hs_feature2d/std_sift/ArrayHelper.h"
#include "hs_feature2d/std_sift/filter.h"


//
namespace hs
{
namespace feature2d
{

typedef hs::imgio::whole::ImageData Image;

class HS_EXPORT ImageHelper: public hs::imgio::whole::ImageIO
{
public:
	//���������ʶ
	enum OprError
	{
		CREATE_IMG_FAIL = -1,
		OPR_OK = 0,
		INVALID_CHANNEL,
		INVALID_SUBSTRACT_IMG
	};

	//ͼ�����ű�ʶ
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

	// RGB��Ҷ�ͼת��
	static int Rgb2Gray(const hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output);

	// ͼ������
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

	//��˹�������
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

	// ��˹ģ��, ����һ��GaussMgr��, ��ʹ����ģ����������˹����ķ�����ʽ����ģ�����ͼ��
	int GaussianBlur(double sigma, int mw, int mh, 
		hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output);

	// �����ڴ���ʾֲ���ԭ���Ż��ĸ�˹ģ��
	int GaussianBlurAccl(double sigma, int mw, int mh,
		hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output);

	// SSE�Ż��ĸ�˹ģ����֤
	int GaussianBlurSSE(double sigma, int mw, int mh,
		hs::imgio::whole::ImageData& img_input, hs::imgio::whole::ImageData& img_output);

private:
	int sseRowFilter(const hs::imgio::whole::ImageData::PByte _src, hs::imgio::whole::ImageData::PByte dst, 
		int w, int cn, int* _kx, int mw);

};




}
}


#endif