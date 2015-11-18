#include <iostream>
#include <cstdint>
#include <cmath>
#include <string>
#include <strstream>

#include "gtest/gtest.h"
#include "hs_feature2d/std_sift/image_helper.hpp"
#include "hs_feature2d/std_sift/std_sift.hpp"

//与opencv对比测试依赖的头文件
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

//ImageHelper类的功能测试
namespace{
	struct CompareImageData
	{
		bool operator() (const hs::imgio::whole::ImageData& image_data1,
			const hs::imgio::whole::ImageData& image_data2,
			unsigned int threshold = 0)
		{
			int width = image_data1.width();
			int height = image_data1.height();
			int number_of_channels = image_data1.channel();
			int bit_depth = image_data1.bit_depth();
			int color_space = image_data1.color_type();

			if (width != image_data2.width() ||
				height != image_data2.height() ||
				number_of_channels != image_data2.channel() ||
				bit_depth != image_data2.bit_depth() ||
				color_space != image_data2.color_type())
			{
				return false;
			}

			uint64_t number_of_samples = 0;
			uint64_t sample_diff = 0;
			if (bit_depth == 1) //深度为1
			{
				int nwidth = (width - 1) / 8 + 1;
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < nwidth; j++)
					{
						for (int k = 0; k < number_of_channels; k++)
						{
							sample_diff +=
								uint64_t(std::abs(int(image_data1.GetByte(i, j, k, 1)) -
								int(image_data2.GetByte(i, j, k, 1))));
						}
					}
				}
				number_of_samples =
					uint64_t(height) * uint64_t(nwidth) * uint64_t(number_of_channels);
			}
			else
			{
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						for (int k = 0; k < number_of_channels; k++)
						{
							sample_diff +=
								uint64_t(std::abs(int(image_data1.GetByte(i, j, k, 1)) -
								int(image_data2.GetByte(i, j, k, 1))));
						}//for (int k = 0; k < number_of_channels; k++)
					}//for (int j = 0; j < width; j++)
				}//for (int i = 0; i < height; i++)
				number_of_samples =
					uint64_t(height) * uint64_t(width) * uint64_t(number_of_channels);
			}//else
			sample_diff /= number_of_samples;
			return sample_diff <= threshold;
		}
	};

	struct ByteCompare
	{
		typedef unsigned char Byte;
		bool operator()(Byte* op1, Byte* op2, size_t w, size_t h, size_t cn)
		{
			int len = h * w * cn, diff = 0, threshold = 0;
			for (int i = 0; i < len; i++)
			{
				diff += std::abs(op1[i] - op2[i]);
			}
			return diff <= threshold;
		}
	};

	TEST(TestImageHelper, Rgb2GrayTest)
	{
		std::string data_path = "../../test_data/";
		std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
		std::string jpeg_gray = "Lenna_gray.jpg";

		hs::imgio::whole::ImageData img_src, img_src2x, img_src1x2, img_gray, img32f, img32g, img8i;
		hs::feature2d::ImageHelper ih;
		int res = 0;

		ASSERT_EQ(0, ih.LoadImage(jpeg_path, img_src));
		res += ih.Rgb2Gray<hs::imgio::whole::ImageData::Byte>(img_src, img_gray);
		ASSERT_EQ(0, res);
		ASSERT_EQ(0, ih.SaveImage(jpeg_gray, img_gray));

		//与OpenCV灰度转换对比
		cv::Mat cvgry, cvsrc = cv::imread(jpeg_path.c_str());
		cv::cvtColor(cvsrc, cvgry, cv::COLOR_RGB2GRAY);

		ByteCompare bcompare;
		ASSERT_EQ(true, bcompare(img_gray.GetBuffer(), cvgry.data, img_gray.width(), img_gray.height(), img_gray.channel() ));
	}

	//灰度转换压测
	TEST(TestImageHelper, Rgb2GrayPress)
	{
		std::string data_path = "../../test_data/";
		std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
		std::string jpeg_gray = "Lenna_gray.jpg";

		hs::feature2d::Mat img_src, img_src2x, img_src1x2, img_gray, img32f, img32g, img8i;
		hs::feature2d::ImageHelper ih;
		ASSERT_EQ(0, ih.LoadImage(jpeg_path, img_src));
		int res = 0;

		clock_t t0, t1;
		t0 = clock();
		for (int i = 0; i < 1000; i++)
		{
			res += ih.Rgb2Gray<hs::imgio::whole::ImageData::Byte>(img_src, img_gray);
		}
		t1 = clock() - t0;
		std::cout << t1 << " ms in gray-scale converting" << std::endl;
		ASSERT_EQ(0, res);
		ASSERT_EQ(0, ih.SaveImage(jpeg_gray, img_gray));
	}

}

