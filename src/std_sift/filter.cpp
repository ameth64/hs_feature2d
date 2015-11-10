#include <algorithm>

#include "hs_feature2d/std_sift/filter.hpp"

namespace hs{
namespace feature2d{





void GaussianFilter::initMask()
{
	bool res = mask_w.Allocate(width_ * 2 + 1) && mask_h.Allocate(height_ * 2 + 1) && imask_w.Allocate(width_ * 2 + 1);
	if (res == false)
	{
		this->state_ = int(GaussianFilter::GsError::CREATE_MASK_FAIL);
	}
	else
		this->state_ = int(GaussianFilter::GsError::GS_OK);

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
}


void GaussianFilter::SetMask(float sx, float sy, int w, int h)
{
	sigmaX_ = (sx > FLT_EPSILON) ? sx : 0.6f;
	sigmaY_ = (sy > FLT_EPSILON) ? sy : sigmaX_;
	width_ = (w > 0) ? w : int(3 * sigmaX_);
	height_ = (h > 0) ? h : int(3 * sigmaY_);
	initMask();
}




}
}