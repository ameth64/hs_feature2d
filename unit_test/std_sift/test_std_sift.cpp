﻿#include <iostream>
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

namespace
{

#define HS_GAUSSIAN_LOOP 10

	//TEST(TestImageHelper, DataConvert_HS_Test)
	//{
	//	std::string data_path = TEST_DATA_PATH;
	//	std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
	//	std::string jpeg_32f = "Lenna_32f.jpg";
	//	
	//	hs::imgio::whole::ImageData img_src, img32f, img8i, img_gray;
	//	hs::feature2d::ImageHelper ih;
	//	ASSERT_EQ(0, ih.LoadImage(jpeg_path, img_src));
	//	ih.Rgb2Gray<hs::imgio::whole::ImageData::Byte, hs::imgio::whole::ImageData::Byte>(img_src, img_gray);

	//	clock_t t0, t1;
	//	int res = 0;
	//	//数据类型转换压测
	//	t0 = clock();
	//	for (int i = 0; i < 1000; i++)
	//	{
	//		ih.ConvertDataType<hs::imgio::whole::ImageData::Byte, float>(img_gray, img32f);
	//	}
	//	t1 = clock() - t0;
	//	std::cout << t1 << " ms in 8u-32f data type converting" << std::endl;
	//	ih.ConvertDataType<float, hs::imgio::whole::ImageData::Byte>(img32f, img8i);
	//	ASSERT_EQ(0, res);
	//	ASSERT_EQ(0, ih.SaveImage(jpeg_32f, img8i));
	//}

	//TEST(TestImageHelper, DataConvert_CV_Test)
	//{
	//	std::string data_path = TEST_DATA_PATH;
	//	std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
	//	std::string jpeg_32f = "Lenna_32f_cv.jpg";

	//	hs::imgio::whole::ImageData img_src, img32f, img8i, img_gray;
	//	hs::feature2d::ImageHelper ih;
	//	ASSERT_EQ(0, ih.LoadImage(jpeg_path, img_src));
	//	ih.Rgb2Gray<hs::imgio::whole::ImageData::Byte, hs::imgio::whole::ImageData::Byte>(img_src, img_gray);

	//	clock_t t0, t1;
	//	int res = 0;
	//	//数据类型转换压测
	//	t0 = clock();
	//	for (int i = 0; i < 1000; i++)
	//	{
	//		ih.ConvertDataType<hs::imgio::whole::ImageData::Byte, float>(img_gray, img32f);
	//	}
	//	t1 = clock() - t0;
	//	std::cout << t1 << " ms in 8u-32f data type converting" << std::endl;
	//	ih.ConvertDataType<float, hs::imgio::whole::ImageData::Byte>(img32f, img8i);
	//	ASSERT_EQ(0, res);
	//	ASSERT_EQ(0, ih.SaveImage(jpeg_32f, img8i));
	//}

	TEST(TestStdSift, GaussianBlur_HS_Test)
	{
		clock_t tt = clock();

		std::string data_path = "../../test_data/";
		std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
		std::string jpeg_gray = "Lenna_gray.jpg"; //灰度输出图片
		std::string jpeg_blr_path = "Lenna_blur.jpg"; //高斯模糊输出图片

		std::cout << "Time cost: " << std::endl;
		hs::imgio::whole::ImageData img_src, img_gray, img_32f, img_8i;
		hs::feature2d::ImageHelper ih;
		ih.LoadImage(jpeg_path, img_src);

		clock_t t0 = clock() - tt, t1;
		std::cout << t0 << " ms in image reading" << std::endl;

		//灰度图转换
		t0 = clock();
		int res = hs::feature2d::ImageHelper::Rgb2Gray<hs::imgio::whole::ImageData::Byte>(img_src, img_gray);
		t1 = clock() - t0;
		std::cout << t1 << " ms in gray-scale converting" << std::endl;

		//res += ih.SaveImage(jpeg_gray, img_gray);

		//数据类型转换
		//ih.ConvertDataType<hs::feature2d::Mat::Byte, float>(img_gray, img_32f);
		ih.ConvertDataType<hs::feature2d::Mat::Byte, float>(img_src, img_32f);

		std::vector< hs::feature2d::Mat > img_blur(HS_GAUSSIAN_LOOP);
		// 循环高斯模糊
		t0 = clock();
		hs::feature2d::GaussianFilter<float, float > gf(1.6f, 4, 4);
		//gf.SetMask(1.6, 0.0f, 4, 4);

		int i = 0, len = HS_GAUSSIAN_LOOP;
		for (; i < len; i++)
		{
			res = gf.Apply(img_32f, img_blur[i]);
		}
		t1 = clock() - t0;
		std::cout << t1 << " ms in " << len << " times bluring process." << std::endl;
		ih.ConvertDataType<float, hs::feature2d::Mat::Byte>(img_blur[0], img_8i);

		//保存图像
		t0 = clock();
		ASSERT_EQ(0, res);
		ASSERT_EQ(0, ih.SaveImage(jpeg_blr_path, img_8i));
		t1 = clock() - t0;
		std::cout << t1 << "ms in image writing" << std::endl;

		tt = clock() - tt;
		std::cout << tt << " ms in whole method call." << std::endl;
	}


	////OpenCV的高斯模糊测试
	//TEST(TestImageHelper, GaussianBlur_OpenCV_Test)
	//{
	//	clock_t tt = clock();
	//	std::string data_path = "../../test_data/";
	//	std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
	//	std::string jpeg_gray_cv_path = "Lenna_gray_cv.jpg"; //opencv库灰度输出图片
	//	std::string jpeg_cv_path = "Lenna_blur_cv.jpg"; //opencv库高斯模糊输出图片


	//	//opencv的cv::GaussianBlur()
	//	std::cout << "Time cost: " << std::endl;
	//	const char* imgPath = "../../test_data/Lenna.jpg";
	//	cv::Mat cv32f, cvres, cvres2, cvgry;
	//	cv::Size ksize(9, 9);
	//	clock_t t0 = clock(), t1 = 0;
	//	cv::Mat cvsrc = cv::imread(jpeg_path.c_str());
	//	t1 = clock() - t0;
	//	std::cout << t1 << "ms in image reading" << std::endl;

	//	//灰度图转换
	//	t0 = clock();
	//	cv::cvtColor(cvsrc, cvgry, cv::COLOR_BGR2GRAY);
	//	t1 = clock() - t0;
	//	std::cout << t1 << " ms in gray-scale converting" << std::endl;

	//	cv::imwrite(jpeg_gray_cv_path, cvgry);

	//	//数据格式转换
	//	cvgry.convertTo(cv32f, CV_32F, 1, 0);

	//	//循环高斯模糊
	//	int i = 0, len = HS_GAUSSIAN_LOOP;
	//	t0 = clock();
	//	
	//	for (; i < len; i++)
	//	{
	//		cv::Mat cvres_;
	//		cv::GaussianBlur(cv32f, cvres_, ksize, 1.6);
	//	}
	//	t1 = clock() - t0;
	//	std::cout << t1 << " ms in " << len << " times bluring process." << std::endl;

	//	cv::GaussianBlur(cv32f, cvres, ksize, 1.6);
	//	//保存图像
	//	t0 = clock();
	//	ASSERT_EQ(true, cv::imwrite(jpeg_cv_path.c_str(), cvres));
	//	t1 = clock() - t0;
	//	std::cout << t1 << "ms in image writing" << std::endl;

	//	tt = clock() - tt;
	//	std::cout << tt << " ms in whole method call." << std::endl;
	//}

	
	/*
	TEST(TestImageHelper, SiftFuncTest)
	{
		std::string data_path = "../../test_data/";
		std::string output_path;

		std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片
		char jpeg_output[512]; //输出图片
		char jpeg_output2[512]; //输出图片

		hs::feature2d::Mat imgsrc, imgBase, imgDesc;
		std::vector<hs::feature2d::KeyPoint> vkp;
		hs::feature2d::ImageHelper ih;
		ih.LoadImage(jpeg_path, imgsrc);
		hs::feature2d::StdSIFT sift;
		
		std::strstream ostr;
		std::vector<hs::feature2d::Mat> gs_pyrmd, dog_pyrmd;

		////测试并输出高斯金字塔及DOG金字塔图像
		//sift.TestFunc(imgsrc, imgBase, gs_pyrmd, dog_pyrmd);
		//output_path = "../../test_data/test_output/Lenna_0_base_image.jpg";
		//imgBase.Convert2Type<hs::feature2d::Mat::Byte, hs::feature2d::SIFT_WORK_TYPE>(imgDesc);
		//ih.SaveImage(output_path, imgDesc);
		//for (int i = 0; i < gs_pyrmd.size(); i++)
		//{
		//	std::sprintf(jpeg_output, "../../test_data/test_output/Lenna_gs_%d_%d_x_%d.jpg", i, gs_pyrmd[i].width(), gs_pyrmd[i].height());
		//	output_path = jpeg_output;
		//	ostr << "../../test_data/test_output/Lenna_gs_" << i << "_" << gs_pyrmd[i].width() << "x" << gs_pyrmd[i].height() << ".jpg";
		//	gs_pyrmd[i].Convert2Type<hs::feature2d::Mat::Byte, hs::feature2d::SIFT_WORK_TYPE>(imgDesc);
		//	ih.SaveImage(output_path, imgDesc);
		//}

		//for (int i = 0; i < dog_pyrmd.size(); i++)
		//{
		//	std::sprintf(jpeg_output2, "../../test_data/test_output/Lenna_dog_%d_%d_x_%d.jpg", i, dog_pyrmd[i].width(), dog_pyrmd[i].height());
		//	output_path = jpeg_output2;
		//	dog_pyrmd[i].Convert2Type<hs::feature2d::Mat::Byte, hs::feature2d::SIFT_WORK_TYPE>(imgDesc);
		//	ih.SaveImage(output_path, imgDesc);
		//}

		// 测试完整的SIFT算法
		sift(imgsrc, vkp, imgDesc);
	}

	TEST(TestImageHelper, OpenCVSiftFuncTest)
	{
		std::string data_path = "../../test_data/";
		std::string output_path;

		std::string jpeg_path = data_path + "Lenna.jpg"; //输入图片

		std::vector<cv::KeyPoint> vkp;
		cv::Mat cvsrc = cv::imread(jpeg_path.c_str());
		cv::Mat cvres, cvdesc, cvmask;
		cv::SIFT sift;
		sift(cvsrc, cvmask, vkp);
		assert(vkp.size() > 0);
	}
	*/
}
