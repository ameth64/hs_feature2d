#ifndef _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_
#define _HS_FEATURE2D_STD_SIFT_IMAGE_HELPER_HPP_

#include <string>
#include <iostream>

#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_image_io/whole_io/image_io.hpp"

//
namespace hs
{
namespace feature2d
{


class HS_EXPORT ImageHelper
{
public:
	ImageHelper();
	ImageHelper(const std::string& imgpath);
	~ImageHelper();

	int LoadImage(const std::string& path);

private:
	std::string img_path_ = "";
	hs::imgio::whole::ImageIO imgio_;
	hs::imgio::whole::ImageData idata_;
};

ImageHelper::ImageHelper()
{
}

ImageHelper::ImageHelper(const std::string& imgpath) : img_path_(imgpath)
{
	imgio_.LoadImage(img_path_, idata_);
}

ImageHelper::~ImageHelper()
{
}


}
}


#endif