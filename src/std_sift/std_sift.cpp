#include <algorithm>
#include "hs_feature2d/std_sift/std_sift.hpp"


namespace hs
{

namespace feature2d
{


const float StdSIFT::SIFT_ORI_PEAK_RATIO = 0.8f;
const float StdSIFT::SIFT_ORI_SIG_FCTR = 1.5f;
const float StdSIFT::SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;
const float StdSIFT::InitSigma = 0.5f;
const float StdSIFT::SIFT_DESCR_SCL_FCTR = 3.f;
const float StdSIFT::SIFT_DESCR_MAG_THR = 0.2f;
const float StdSIFT::SIFT_INT_DESCR_FCTR = 512.0f;

StdSIFT::~StdSIFT()
{}

StdSIFT::StdSIFT(int nf /*= 0*/, int nol /*= 3*/, double ct /*= 0.04*/, double et /*= 10*/, double s /*= 1.6*/)
	: i_features(nf), i_octave_layer(nol), contrast_threshold(ct), edge_threshold(et), sigma(s)
{}


int StdSIFT::createInitialImage(Image& src, Image& dst, bool doubleImageSize, float sigma)
{
	int res = 0;
	Image gray, gray_fpt;
	if (src.channel() == 3 || src.channel() == 4)
		res += hs::feature2d::ImageHelper::Rgb2Gray<Image::Byte, Image::Byte>(src, gray);
	else
		gray.DeepCopy(src);
	res += gray.Convert2Type<SIFT_WORK_TYPE>(gray_fpt);
	
	float sig_diff;
	if (doubleImageSize)
	{
		sig_diff = sqrtf(std::max(sigma * sigma - InitSigma * InitSigma * 4, 0.01f));
		hs::feature2d::GaussianFilter<SIFT_WORK_TYPE, SIFT_WORK_TYPE > gsfilter(sig_diff);
		Image dbl;
		hs::feature2d::ImageHelper::Resize<SIFT_WORK_TYPE>(gray_fpt, dbl, gray_fpt.width() * 2, gray_fpt.height() * 2, hs::feature2d::ImageHelper::INTER_LINEAR);
		//GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
		res += gsfilter.Apply(dbl, dst);
		//return dbl;
	}
	else
	{
		sig_diff = sqrtf(std::max(sigma * sigma - InitSigma * InitSigma, 0.01f));
		hs::feature2d::GaussianFilter<SIFT_WORK_TYPE, SIFT_WORK_TYPE > gsfilter(sig_diff);
		//GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
		res += gsfilter.Apply(gray_fpt, dst);
		//return gray_fpt;
	}
	return res;
}




int StdSIFT::operator()(Image& img, std::vector<KeyPoint>& keypoints, Image& descriptors, bool useProvidedKeypoints /*= false*/)
{
	// 金字塔组初始参数
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	// 创建初始图像
	Image baseImg;
	int res = createInitialImage(img, baseImg, firstOctave < 0, (float)sigma);
	std::vector<Image> gs_pyrmd, dog_pyrmd;

	//设定层数
	//int nOctaves = actualNOctaves > 0 ? actualNOctaves : sseRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;
	int nOctaves = sseRound( log( (double)std::min(baseImg.width(), baseImg.height()) ) / log(2.0) - 2.0 ) - firstOctave;

	//buildGaussianPyramid(base, gpyr, nOctaves);
	//buildDoGPyramid(gpyr, dogpyr);
	BuildGaussianPyramid(baseImg, gs_pyrmd, nOctaves);
	BuildDoGPyramid(gs_pyrmd, dog_pyrmd);

	//
	//t = (double)getTickCount();
	//findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
	FindScaleSpaceExtrema(gs_pyrmd, dog_pyrmd, keypoints);
	KeyPointsFilter::removeDuplicated(keypoints);

	if (i_features > 0)
		KeyPointsFilter::retainBest(keypoints, i_features);
	//t = (double)getTickCount() - t;
	//printf("keypoint detection time: %g\n", t*1000./tf);

	if (firstOctave < 0){
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint& kpt = keypoints[i];
			float scale = 1.f / (float)(1 << -firstOctave);
			kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			kpt.pt *= scale;
			kpt.size *= scale;
		}
	}


	//t = (double)getTickCount();
	int dsize = descriptorSize();
	descriptors.CreateImage(dsize, (int)keypoints.size(), 1, 32);
	//Mat descriptors = _descriptors.getMat();

	CalcDescriptors(gs_pyrmd, keypoints, descriptors, i_octave_layer, firstOctave);
	//t = (double)getTickCount() - t;
	//printf("descriptor extraction time: %g\n", t*1000./tf);
	return res;
}

int StdSIFT::operator()(Image& img, Image& mask, std::vector<KeyPoint>& keypoints, Image& descriptors, bool useProvidedKeypoints /*= false*/) const
{
	return operator()(img, mask, keypoints, descriptors, useProvidedKeypoints);
}


void StdSIFT::BuildGaussianPyramid(const Image& base, std::vector<Image>& pyr, int nOctaves) const
{
	std::vector<double> sig( i_octave_layer + 3 );
	pyr.resize(nOctaves*(i_octave_layer + 3));

	// 按以下公式计算各层的尺度:
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	sig[0] = sigma;
	double k = pow(2., 1. / i_octave_layer);
	for (int i = 1; i < i_octave_layer + 3; i++)
	{
		double sig_prev = pow(k, (double)(i - 1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	hs::feature2d::GaussianFilter<SIFT_WORK_TYPE, SIFT_WORK_TYPE > gsf;
	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < i_octave_layer + 3; i++)
		{
			Image& dst = pyr[o*(i_octave_layer + 3) + i];
			if (o == 0 && i == 0)
				dst = base;
			// base of new octave is halved image from end of previous octave
			else if (i == 0)
			{
				const Image& src = pyr[(o - 1)*(i_octave_layer + 3) + i_octave_layer];
				//resize(src, dst, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);
				hs::feature2d::ImageHelper::Resize<SIFT_WORK_TYPE>(src, dst, src.width() / 2, src.height() / 2, 
					hs::feature2d::ImageHelper::INTER_NEAREST);
			}
			else
			{
				const Image& src = pyr[o*(i_octave_layer + 3) + i - 1];
				//GaussianBlur(src, dst, Size(), sig[i], sig[i]);
				gsf.SetMask(sig[i]);
				gsf.Apply(src, dst);
			}
		}
	}
}


// 注意为保证精度, DOG金字塔需用float类型存储
void StdSIFT::BuildDoGPyramid(const std::vector<Image>& gpyr, std::vector<Image>& dogpyr) const
{
	int nOctaves = (int)gpyr.size() / (this->i_octave_layer + 3);
	dogpyr.resize(nOctaves*(i_octave_layer + 2));

	int res = 0;
	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < i_octave_layer + 2; i++)
		{
			const Image& src1 = gpyr[o*(i_octave_layer + 3) + i];
			const Image& src2 = gpyr[o*(i_octave_layer + 3) + i + 1];
			Image& dst = dogpyr[o*(i_octave_layer + 2) + i];
			//subtract(src2, src1, dst, noArray(), DataType<SIFT_WORK_TYPE>::type);
			res += hs::feature2d::ImageHelper::Subtract<SIFT_WORK_TYPE>(src1, src2, dst);
		}
	}
}

//定位局部极值点
void StdSIFT::FindScaleSpaceExtrema(const std::vector<Image>& gauss_pyr, const std::vector<Image>& dog_pyr, 
	std::vector<KeyPoint>& keypoints) const
{
	int nOctaves = (int)gauss_pyr.size() / (i_octave_layer + 3);
	int threshold = sseFloor(0.5 * contrast_threshold / i_octave_layer * 255);
	const int n = SIFT_ORI_HIST_BINS;
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();
	
	for (int o = 0; o < nOctaves; o++)
		for (int i = 1; i <= i_octave_layer; i++)
		{
			int idx = o*(i_octave_layer + 2) + i;
			const Image& img = dog_pyr[idx];
			const Image& prev = dog_pyr[idx - 1];
			const Image& next = dog_pyr[idx + 1];
			//int step = (int)img.step1();
			int rows = img.height(), cols = img.width();
			int step = cols*img.channel();

			for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++)
			{
				const SIFT_WORK_TYPE* currptr = (SIFT_WORK_TYPE*)img.GetLine(r);
				const SIFT_WORK_TYPE* prevptr = (SIFT_WORK_TYPE*)prev.GetLine(r);
				const SIFT_WORK_TYPE* nextptr = (SIFT_WORK_TYPE*)next.GetLine(r);

				for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
				{
					SIFT_WORK_TYPE val = currptr[c];

					// find local extrema with pixel accuracy
					if (std::abs(val) > threshold &&
						((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
						val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
						val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
						val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
						val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
						val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
						val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
						val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
						val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
						(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
						val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
						val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
						val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
						val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
						val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
						val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
						val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
						val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
					{
						int r1 = r, c1 = c, layer = i;
						if (!AdjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
							i_octave_layer, (float)contrast_threshold,
							(float)edge_threshold, (float)sigma))
							continue;
						float scl_octv = kpt.size*0.5f / (1 << o);
						//float omax = calcOrientationHist(gauss_pyr[o*(i_octave_layer + 3) + layer],
						float omax = CalcOrientationHist(gauss_pyr[o*(i_octave_layer + 3) + layer],
							Point(c1, r1),
							sseRound(SIFT_ORI_RADIUS * scl_octv),
							SIFT_ORI_SIG_FCTR * scl_octv,
							hist, n);
						float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
						for (int j = 0; j < n; j++)
						{
							int l = j > 0 ? j - 1 : n - 1;
							int r2 = j < n - 1 ? j + 1 : 0;

							if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
							{
								float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = 360.f - (float)((360.f / n) * bin);
								if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
									kpt.angle = 0.f;
								keypoints.push_back(kpt);
							}
						}
						//
					}
				}
			}
		}
		
}

//调整过滤局部极值点
bool StdSIFT::AdjustLocalExtrema(const std::vector<Image>& dog_pyr, KeyPoint& kpt, int octv, int& layer, int& r, int& c, 
	int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma)
{
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	const float deriv_scale = img_scale*0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale*0.25f;

	float xi = 0, xr = 0, xc = 0, contr = 0;
	int i = 0;

	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		int idx = octv*(nOctaveLayers + 2) + layer;
		const Image& img = dog_pyr[idx];
		const Image& prev = dog_pyr[idx - 1];
		const Image& next = dog_pyr[idx + 1];

		Vec3f dD(
			(img.GetElem<SIFT_WORK_TYPE>(r, c + 1) - img.GetElem<SIFT_WORK_TYPE>(r, c - 1))*deriv_scale,
			(img.GetElem<SIFT_WORK_TYPE>(r + 1, c) - img.GetElem<SIFT_WORK_TYPE>(r - 1, c))*deriv_scale,
			(next.GetElem<SIFT_WORK_TYPE>(r, c) - prev.GetElem<SIFT_WORK_TYPE>(r, c))*deriv_scale
		);

		float v2 = (float)img.GetElem<SIFT_WORK_TYPE>(r, c) * 2;
		float dxx = (img.GetElem<SIFT_WORK_TYPE>(r, c + 1) + img.GetElem<SIFT_WORK_TYPE>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.GetElem<SIFT_WORK_TYPE>(r + 1, c) + img.GetElem<SIFT_WORK_TYPE>(r - 1, c) - v2)*second_deriv_scale;
		float dss = (next.GetElem<SIFT_WORK_TYPE>(r, c) + prev.GetElem<SIFT_WORK_TYPE>(r, c) - v2)*second_deriv_scale;
		float dxy = (img.GetElem<SIFT_WORK_TYPE>(r + 1, c + 1) - img.GetElem<SIFT_WORK_TYPE>(r + 1, c - 1) -
			img.GetElem<SIFT_WORK_TYPE>(r - 1, c + 1) + img.GetElem<SIFT_WORK_TYPE>(r - 1, c - 1))*cross_deriv_scale;
		float dxs = (next.GetElem<SIFT_WORK_TYPE>(r, c + 1) - next.GetElem<SIFT_WORK_TYPE>(r, c - 1) -
			prev.GetElem<SIFT_WORK_TYPE>(r, c + 1) + prev.GetElem<SIFT_WORK_TYPE>(r, c - 1))*cross_deriv_scale;
		float dys = (next.GetElem<SIFT_WORK_TYPE>(r + 1, c) - next.GetElem<SIFT_WORK_TYPE>(r - 1, c) -
			prev.GetElem<SIFT_WORK_TYPE>(r + 1, c) + prev.GetElem<SIFT_WORK_TYPE>(r - 1, c))*cross_deriv_scale;

		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);
		//Vec3f X = solve(H, dD, DECOMP_LU);

		xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
			break;

		if (std::abs(xi) > (float)(INT_MAX / 3) ||
			std::abs(xr) > (float)(INT_MAX / 3) ||
			std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		c += sseRound(xc);
		r += sseRound(xr);
		layer += sseRound(xi);

		if (layer < 1 || layer > nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.width() - SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER || r >= img.height() - SIFT_IMG_BORDER)
			return false;
	}

	// ensure convergence of interpolation
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	{
		int idx = octv*(nOctaveLayers + 2) + layer;
		const Image& img = dog_pyr[idx];
		const Image& prev = dog_pyr[idx - 1];
		const Image& next = dog_pyr[idx + 1];
		Matx31f dD((img.GetElem<SIFT_WORK_TYPE>(r, c + 1) - img.GetElem<SIFT_WORK_TYPE>(r, c - 1))*deriv_scale,
			(img.GetElem<SIFT_WORK_TYPE>(r + 1, c) - img.GetElem<SIFT_WORK_TYPE>(r - 1, c))*deriv_scale,
			(next.GetElem<SIFT_WORK_TYPE>(r, c) - prev.GetElem<SIFT_WORK_TYPE>(r, c))*deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));

		contr = img.GetElem<SIFT_WORK_TYPE>(r, c)*img_scale + t * 0.5f;
		if (std::abs(contr) * nOctaveLayers < contrastThreshold)
			return false;

		// principal curvatures are computed using the trace and det of Hessian
		float v2 = img.GetElem<SIFT_WORK_TYPE>(r, c)*2.f;
		float dxx = (img.GetElem<SIFT_WORK_TYPE>(r, c + 1) + img.GetElem<SIFT_WORK_TYPE>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.GetElem<SIFT_WORK_TYPE>(r + 1, c) + img.GetElem<SIFT_WORK_TYPE>(r - 1, c) - v2)*second_deriv_scale;
		float dxy = (img.GetElem<SIFT_WORK_TYPE>(r + 1, c + 1) - img.GetElem<SIFT_WORK_TYPE>(r + 1, c - 1) -
			img.GetElem<SIFT_WORK_TYPE>(r - 1, c + 1) + img.GetElem<SIFT_WORK_TYPE>(r - 1, c - 1)) * cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (sseRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
	kpt.response = std::abs(contr);

	return true;
}

void StdSIFT::CalcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave)
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint kpt = keypoints[i];
		int octave, layer;
		float scale;
		unpackOctave(kpt, octave, layer, scale);
		assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
		float size = kpt.size*scale;
		Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
		const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

		float angle = 360.f - kpt.angle;
		if (std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;
		calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.GetLineT<float>((int)i));
	}
}

float StdSIFT::CalcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n)
{
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	//AutoBuffer<float> buf(len * 4 + n + 4);
	AutoBuffer buf(len * 4 + n + 4);
	
	float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
	float* temphist = W + len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.height() - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.width() - 1)
				continue;

			float dx = (float)(img.GetElem<SIFT_WORK_TYPE>(y, x + 1) - img.GetElem<SIFT_WORK_TYPE>(y, x - 1));
			float dy = (float)(img.GetElem<SIFT_WORK_TYPE>(y - 1, x) - img.GetElem<SIFT_WORK_TYPE>(y + 1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	hs::feature2d::exp(W, W, len);
	hs::feature2d::fastAtan2(Y, X, Ori, len, true);
	hs::feature2d::magnitude(X, Y, Mag, len);

	for (k = 0; k < len; k++)
	{
		int bin = sseRound((n / 360.f)*Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = std::max(maxval, hist[i]);

	return maxval;
}

void StdSIFT::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst)

{
	Point pt(sseRound(ptf.x), sseRound(ptf.y));
	float cos_t = cosf(ori*(float)(HS_PI / 180));
	float sin_t = sinf(ori*(float)(HS_PI / 180));
	float bins_per_rad = n / 360.f;
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = sseRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt((double)img.width()*img.width() + img.height()*img.height()));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.height(), cols = img.width();

	AutoBuffer buf(len * 6 + histlen);
	float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
		// Calculate sample's histogram array coords rotated relative to ori.
		// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
		// r_rot = 1.5) have full weight placed in row 1 after interpolation.
		float c_rot = j * cos_t - i * sin_t;
		float r_rot = j * sin_t + i * cos_t;
		float rbin = r_rot + d / 2 - 0.5f;
		float cbin = c_rot + d / 2 - 0.5f;
		int r = pt.y + i, c = pt.x + j;

		if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
			r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
		{
			float dx = (float)(img.GetElem<SIFT_WORK_TYPE>(r, c + 1) - img.GetElem<SIFT_WORK_TYPE>(r, c - 1));
			float dy = (float)(img.GetElem<SIFT_WORK_TYPE>(r - 1, c) - img.GetElem<SIFT_WORK_TYPE>(r + 1, c));
			X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
			W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
			k++;
		}
		}

	len = k;
	hs::feature2d::fastAtan2(Y, X, Ori, len, true);
	hs::feature2d::magnitude(X, Y, Mag, len);
	hs::feature2d::exp(W, W, len);

	for (k = 0; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		float obin = (Ori[k] - ori)*bins_per_rad;
		float mag = Mag[k] * W[k];

		int r0 = sseFloor(rbin);
		int c0 = sseFloor(cbin);
		int o0 = sseFloor(obin);
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		if (o0 < 0)
			o0 += n;
		if (o0 >= n)
			o0 -= n;

		// histogram update using tri-linear interpolation
		float v_r1 = mag*rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
		int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
		hist[idx] += hist[idx + n];
		hist[idx + 1] += hist[idx + n + 1];
		for (k = 0; k < n; k++)
			dst[(i*d + j)*n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d*d*n;
	for (k = 0; k < len; k++)
		nrm2 += dst[k] * dst[k];
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
	for (k = 0; k < len; k++)
	{
		dst[k] = saturate_cast<Mat::Byte>(dst[k] * nrm2);
	}
#else
	float nrm1 = 0;
	for (k = 0; k < len; k++)
	{
		dst[k] *= nrm2;
		nrm1 += dst[k];
	}
	nrm1 = 1.f / std::max(nrm1, FLT_EPSILON);
	for (k = 0; k < len; k++)
	{
		dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
	}
#endif
}

void StdSIFT::unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

int StdSIFT::descriptorSize() const
{
	return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int StdSIFT::TestFunc(hs::feature2d::Mat& img, hs::feature2d::Mat& baseImg, std::vector<Image>& gs_pyrmd, std::vector<Image>& dog_pyrmd)
{
	// 金字塔组初始参数
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	// 创建初始图像
	int res = createInitialImage(img, baseImg, firstOctave < 0, (float)sigma);

	//设定层数
	//int nOctaves = actualNOctaves > 0 ? actualNOctaves : sseRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;
	int nOctaves = sseRound(log((double)std::min(baseImg.width(), baseImg.height())) / log(2.0) - 2.0) - firstOctave;

	//buildGaussianPyramid(base, gpyr, nOctaves);
	//buildDoGPyramid(gpyr, dogpyr);
	BuildGaussianPyramid(baseImg, gs_pyrmd, nOctaves);
	BuildDoGPyramid(gs_pyrmd, dog_pyrmd);
	return 0;
}





}

}
