#ifndef _HS_FEATURE2D_STD_SIFT_STD_SIFT_HPP_
#define _HS_FEATURE2D_STD_SIFT_STD_SIFT_HPP_

#include <vector>
//#include "hs_feature2d//std_sift/matrix.hpp"
//#include "hs_feature2d/std_sift/filter.hpp"
#include "hs_feature2d/std_sift/image_helper.hpp"


namespace hs
{

namespace feature2d
{

typedef float SIFT_WORK_TYPE;
typedef hs::feature2d::HeapMgr<SIFT_WORK_TYPE> AutoBuffer;
typedef hs::imgio::whole::ImageData Image;



//��׼SIFT�㷨
class HS_EXPORT StdSIFT
{
public:
	static const float InitSigma;

	//Ĭ�Ϸ���ֱ��ͼ������
	static const int SIFT_ORI_HIST_BINS = 36;
	//����ؼ���ʱ���Եı߽���
	static const int SIFT_IMG_BORDER = 5;
	static const int SIFT_FIXPT_SCALE = 1;
	// maximum steps of keypoint interpolation before failure
	static const int SIFT_MAX_INTERP_STEPS = 5;

	// default width of descriptor histogram array
	static const int SIFT_DESCR_WIDTH = 4;

	// default number of bins per histogram in descriptor array
	static const int SIFT_DESCR_HIST_BINS = 8;

	// orientation magnitude relative to max that results in new feature
	static const float SIFT_ORI_PEAK_RATIO;
	// determines gaussian sigma for orientation assignment
	static const float SIFT_ORI_SIG_FCTR;
	// determines the radius of the region used in orientation assignment
	static const float SIFT_ORI_RADIUS;

	static const float SIFT_DESCR_SCL_FCTR;
	static const float SIFT_DESCR_MAG_THR;
	static const float SIFT_INT_DESCR_FCTR;

	StdSIFT(int nf = 0, int nol = 3, double ct = 0.04, double et = 10, double s = 1.6);
	~StdSIFT();

	int operator()(Image& img, std::vector<KeyPoint>& keypoints,
		Image& descriptors,
		bool useProvidedKeypoints = false);

	int operator()(Image& img, Image& mask,
		std::vector<KeyPoint>& keypoints,
		Image& descriptors,
		bool useProvidedKeypoints = false) const;

	// ������˹������
	void BuildGaussianPyramid(const Image& base, std::vector<Image>& pyr, int nOctaves) const;
	// ����DOG������
	void BuildDoGPyramid(const std::vector<Image>& gpyr, std::vector<Image>& dogpyr) const;

	//Ѱ�Ҽ�ֵ��
	void FindScaleSpaceExtrema(const std::vector<Image>& gauss_pyr, const std::vector<Image>& dog_pyr,
		std::vector<KeyPoint>& keypoints) const;
	// ���˾ֲ���ֵ��
	static bool AdjustLocalExtrema(const std::vector<Image>& dog_pyr, KeyPoint& kpt, int octv,
		int& layer, int& r, int& c, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma);
	// ����������
	static void CalcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
		Mat& descriptors, int nOctaveLayers, int firstOctave);

	static void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst);

	static float CalcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n);

	static inline void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale);

	int inline descriptorSize() const;
	
	// ���Ը�˹��������DOG���������
	int TestFunc(hs::feature2d::Mat& img, hs::feature2d::Mat& baseImg, std::vector<Image>& gs_pyrmd, std::vector<Image>& dog_pyrmd);
private:
	int createInitialImage(Image& src, Image& dst, bool doubleImageSize, float sigma);

	int i_features;
	int i_octave_layer;
	double contrast_threshold;
	double edge_threshold;
	double sigma;
};

StdSIFT::~StdSIFT()
{}




}

}



#endif //#_HS_FEATURE2D_STD_SIFT_STD_SIFT_HPP_