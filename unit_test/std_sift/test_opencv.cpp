#include <iostream>
#include <cstdint>
#include <cmath>
#include <string>
#include <strstream>

#include "gtest/gtest.h"
#include "hs_feature2d/std_sift/image_helper.hpp"
#include "hs_feature2d/std_sift/std_sift.hpp"

//��opencv�ԱȲ���������ͷ�ļ�
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

namespace
{

#define CV_GAUSSIAN_LOOP 10

	//OpenCV�ĸ�˹ģ������
	TEST(TestOpenCV, GaussianBlur_OpenCV_Test)
	{
		clock_t tt = clock();
		std::string data_path = "../../test_data/";
		std::string jpeg_path = data_path + "Lenna.jpg"; //����ͼƬ
		std::string jpeg_gray_cv_path = "Lenna_gray_cv.jpg"; //opencv��Ҷ����ͼƬ
		std::string jpeg_cv_path = "Lenna_blur_cv.jpg"; //opencv���˹ģ�����ͼƬ


		//opencv��cv::GaussianBlur()
		std::cout << "Time cost: " << std::endl;
		const char* imgPath = "../../test_data/Lenna.jpg";
		cv::Mat cv32f, cvres, cvres2, cvgry;
		cv::Size ksize(9, 9);
		clock_t t0 = clock(), t1 = 0;
		cv::Mat cvsrc = cv::imread(jpeg_path.c_str());
		t1 = clock() - t0;
		std::cout << t1 << "ms in image reading" << std::endl;

		//�Ҷ�ͼת��
		t0 = clock();
		cv::cvtColor(cvsrc, cvgry, cv::COLOR_RGB2GRAY);
		t1 = clock() - t0;
		std::cout << t1 << " ms in gray-scale converting" << std::endl;

		cv::imwrite(jpeg_gray_cv_path, cvgry);

		//���ݸ�ʽת��
		//cvgry.convertTo(cv32f, CV_32F, 1, 0);
		cvsrc.convertTo(cv32f, CV_32F, 1, 0);

		std::vector< cv::Mat > cvres_(CV_GAUSSIAN_LOOP);
		//ѭ����˹ģ��
		int i = 0, len = CV_GAUSSIAN_LOOP;
		t0 = clock();
		for (; i < len; i++)
		{
			cv::GaussianBlur(cv32f, cvres_[i], ksize, 1.6+float(i/CV_GAUSSIAN_LOOP) );
		}
		t1 = clock() - t0;
		std::cout << t1 << " ms in " << len << " times bluring process." << std::endl;

		cvres_[0].convertTo(cvres, CV_8U, 1, 0);
		//����ͼ��
		t0 = clock();
		ASSERT_EQ(true, cv::imwrite(jpeg_cv_path.c_str(), cvres));
		t1 = clock() - t0;
		std::cout << t1 << "ms in image writing" << std::endl;

		tt = clock() - tt;
		std::cout << tt << " ms in whole method call." << std::endl;
	}


	/*
	TEST(TestImageHelper, SiftFuncTest)
	{
	std::string data_path = "../../test_data/";
	std::string output_path;

	std::string jpeg_path = data_path + "Lenna.jpg"; //����ͼƬ
	char jpeg_output[512]; //���ͼƬ
	char jpeg_output2[512]; //���ͼƬ

	hs::feature2d::Mat imgsrc, imgBase, imgDesc;
	std::vector<hs::feature2d::KeyPoint> vkp;
	hs::feature2d::ImageHelper ih;
	ih.LoadImage(jpeg_path, imgsrc);
	hs::feature2d::StdSIFT sift;

	std::strstream ostr;
	std::vector<hs::feature2d::Mat> gs_pyrmd, dog_pyrmd;

	////���Բ������˹��������DOG������ͼ��
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

	// ����������SIFT�㷨
	sift(imgsrc, vkp, imgDesc);
	}

	TEST(TestImageHelper, OpenCVSiftFuncTest)
	{
	std::string data_path = "../../test_data/";
	std::string output_path;

	std::string jpeg_path = data_path + "Lenna.jpg"; //����ͼƬ

	std::vector<cv::KeyPoint> vkp;
	cv::Mat cvsrc = cv::imread(jpeg_path.c_str());
	cv::Mat cvres, cvdesc, cvmask;
	cv::SIFT sift;
	sift(cvsrc, cvmask, vkp);
	assert(vkp.size() > 0);
	}
	*/
}
