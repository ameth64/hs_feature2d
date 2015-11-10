#include <algorithm>

#include "hs_feature2d/std_sift/numeric_solver.hpp"

namespace hs{
namespace feature2d{

static void Exp_32f(const float *_x, float *y, int n)
{
	static const float
		A4 = (float)(1.000000000000002438532970795181890933776 / EXPPOLY_32F_A0),
		A3 = (float)(.6931471805521448196800669615864773144641 / EXPPOLY_32F_A0),
		A2 = (float)(.2402265109513301490103372422686535526573 / EXPPOLY_32F_A0),
		A1 = (float)(.5550339366753125211915322047004666939128e-1 / EXPPOLY_32F_A0);

#undef EXPPOLY
#define EXPPOLY(x)  \
    (((((x) + A1)*(x) + A2)*(x) + A3)*(x) + A4)

	int i = 0;
	const Hs32suf* x = (const Hs32suf*)_x;
	Hs32suf buf[4];

#if _USE_SSE2_
	if (n >= 8)
	{
		static const __m128d prescale2 = _mm_set1_pd(exp_prescale);
		static const __m128 postscale4 = _mm_set1_ps((float)exp_postscale);
		static const __m128 maxval4 = _mm_set1_ps((float)(exp_max_val / exp_prescale));
		static const __m128 minval4 = _mm_set1_ps((float)(-exp_max_val / exp_prescale));

		static const __m128 mA1 = _mm_set1_ps(A1);
		static const __m128 mA2 = _mm_set1_ps(A2);
		static const __m128 mA3 = _mm_set1_ps(A3);
		static const __m128 mA4 = _mm_set1_ps(A4);
		bool y_aligned = (size_t)(void*)y % 16 == 0;

		ushort _DECL_ALIGNED(16) tab_idx[8];

		for (; i <= n - 8; i += 8)
		{
			__m128 xf0, xf1;
			xf0 = _mm_loadu_ps(&x[i].f);
			xf1 = _mm_loadu_ps(&x[i + 4].f);
			__m128i xi0, xi1, xi2, xi3;

			xf0 = _mm_min_ps(_mm_max_ps(xf0, minval4), maxval4);
			xf1 = _mm_min_ps(_mm_max_ps(xf1, minval4), maxval4);

			__m128d xd0 = _mm_cvtps_pd(xf0);
			__m128d xd2 = _mm_cvtps_pd(_mm_movehl_ps(xf0, xf0));
			__m128d xd1 = _mm_cvtps_pd(xf1);
			__m128d xd3 = _mm_cvtps_pd(_mm_movehl_ps(xf1, xf1));

			xd0 = _mm_mul_pd(xd0, prescale2);
			xd2 = _mm_mul_pd(xd2, prescale2);
			xd1 = _mm_mul_pd(xd1, prescale2);
			xd3 = _mm_mul_pd(xd3, prescale2);

			xi0 = _mm_cvtpd_epi32(xd0);
			xi2 = _mm_cvtpd_epi32(xd2);

			xi1 = _mm_cvtpd_epi32(xd1);
			xi3 = _mm_cvtpd_epi32(xd3);

			xd0 = _mm_sub_pd(xd0, _mm_cvtepi32_pd(xi0));
			xd2 = _mm_sub_pd(xd2, _mm_cvtepi32_pd(xi2));
			xd1 = _mm_sub_pd(xd1, _mm_cvtepi32_pd(xi1));
			xd3 = _mm_sub_pd(xd3, _mm_cvtepi32_pd(xi3));

			xf0 = _mm_movelh_ps(_mm_cvtpd_ps(xd0), _mm_cvtpd_ps(xd2));
			xf1 = _mm_movelh_ps(_mm_cvtpd_ps(xd1), _mm_cvtpd_ps(xd3));

			xf0 = _mm_mul_ps(xf0, postscale4);
			xf1 = _mm_mul_ps(xf1, postscale4);

			xi0 = _mm_unpacklo_epi64(xi0, xi2);
			xi1 = _mm_unpacklo_epi64(xi1, xi3);
			xi0 = _mm_packs_epi32(xi0, xi1);

			_mm_store_si128((__m128i*)tab_idx, _mm_and_si128(xi0, _mm_set1_epi16(EXPTAB_MASK)));

			xi0 = _mm_add_epi16(_mm_srai_epi16(xi0, EXPTAB_SCALE), _mm_set1_epi16(127));
			xi0 = _mm_max_epi16(xi0, _mm_setzero_si128());
			xi0 = _mm_min_epi16(xi0, _mm_set1_epi16(255));
			xi1 = _mm_unpackhi_epi16(xi0, _mm_setzero_si128());
			xi0 = _mm_unpacklo_epi16(xi0, _mm_setzero_si128());

			__m128d yd0 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[0]), _mm_load_sd(expTab + tab_idx[1]));
			__m128d yd1 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[2]), _mm_load_sd(expTab + tab_idx[3]));
			__m128d yd2 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[4]), _mm_load_sd(expTab + tab_idx[5]));
			__m128d yd3 = _mm_unpacklo_pd(_mm_load_sd(expTab + tab_idx[6]), _mm_load_sd(expTab + tab_idx[7]));

			__m128 yf0 = _mm_movelh_ps(_mm_cvtpd_ps(yd0), _mm_cvtpd_ps(yd1));
			__m128 yf1 = _mm_movelh_ps(_mm_cvtpd_ps(yd2), _mm_cvtpd_ps(yd3));

			yf0 = _mm_mul_ps(yf0, _mm_castsi128_ps(_mm_slli_epi32(xi0, 23)));
			yf1 = _mm_mul_ps(yf1, _mm_castsi128_ps(_mm_slli_epi32(xi1, 23)));

			__m128 zf0 = _mm_add_ps(xf0, mA1);
			__m128 zf1 = _mm_add_ps(xf1, mA1);

			zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA2);
			zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA2);

			zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA3);
			zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA3);

			zf0 = _mm_add_ps(_mm_mul_ps(zf0, xf0), mA4);
			zf1 = _mm_add_ps(_mm_mul_ps(zf1, xf1), mA4);

			zf0 = _mm_mul_ps(zf0, yf0);
			zf1 = _mm_mul_ps(zf1, yf1);

			if (y_aligned)
			{
				_mm_store_ps(y + i, zf0);
				_mm_store_ps(y + i + 4, zf1);
			}
			else
			{
				_mm_storeu_ps(y + i, zf0);
				_mm_storeu_ps(y + i + 4, zf1);
			}
		}
	}
	else
#endif
		for (; i <= n - 4; i += 4)
		{
		double x0 = x[i].f * exp_prescale;
		double x1 = x[i + 1].f * exp_prescale;
		double x2 = x[i + 2].f * exp_prescale;
		double x3 = x[i + 3].f * exp_prescale;
		int val0, val1, val2, val3, t;

		if (((x[i].i >> 23) & 255) > 127 + 10)
			x0 = x[i].i < 0 ? -exp_max_val : exp_max_val;

		if (((x[i + 1].i >> 23) & 255) > 127 + 10)
			x1 = x[i + 1].i < 0 ? -exp_max_val : exp_max_val;

		if (((x[i + 2].i >> 23) & 255) > 127 + 10)
			x2 = x[i + 2].i < 0 ? -exp_max_val : exp_max_val;

		if (((x[i + 3].i >> 23) & 255) > 127 + 10)
			x3 = x[i + 3].i < 0 ? -exp_max_val : exp_max_val;

		val0 = sseRound(x0);
		val1 = sseRound(x1);
		val2 = sseRound(x2);
		val3 = sseRound(x3);

		x0 = (x0 - val0)*exp_postscale;
		x1 = (x1 - val1)*exp_postscale;
		x2 = (x2 - val2)*exp_postscale;
		x3 = (x3 - val3)*exp_postscale;

		t = (val0 >> EXPTAB_SCALE) + 127;
		t = !(t & ~255) ? t : t < 0 ? 0 : 255;
		buf[0].i = t << 23;

		t = (val1 >> EXPTAB_SCALE) + 127;
		t = !(t & ~255) ? t : t < 0 ? 0 : 255;
		buf[1].i = t << 23;

		t = (val2 >> EXPTAB_SCALE) + 127;
		t = !(t & ~255) ? t : t < 0 ? 0 : 255;
		buf[2].i = t << 23;

		t = (val3 >> EXPTAB_SCALE) + 127;
		t = !(t & ~255) ? t : t < 0 ? 0 : 255;
		buf[3].i = t << 23;

		x0 = buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY(x0);
		x1 = buf[1].f * expTab[val1 & EXPTAB_MASK] * EXPPOLY(x1);

		y[i] = (float)x0;
		y[i + 1] = (float)x1;

		x2 = buf[2].f * expTab[val2 & EXPTAB_MASK] * EXPPOLY(x2);
		x3 = buf[3].f * expTab[val3 & EXPTAB_MASK] * EXPPOLY(x3);

		y[i + 2] = (float)x2;
		y[i + 3] = (float)x3;
		}

	for (; i < n; i++)
	{
		double x0 = x[i].f * exp_prescale;
		int val0, t;

		if (((x[i].i >> 23) & 255) > 127 + 10)
			x0 = x[i].i < 0 ? -exp_max_val : exp_max_val;

		val0 = sseRound(x0);
		t = (val0 >> EXPTAB_SCALE) + 127;
		t = !(t & ~255) ? t : t < 0 ? 0 : 255;

		buf[0].i = t << 23;
		x0 = (x0 - val0)*exp_postscale;

		y[i] = (float)(buf[0].f * expTab[val0 & EXPTAB_MASK] * EXPPOLY(x0));
	}
}

void exp(const float* src, float* dst, int n)
{
	Exp_32f(src, dst, n);
}


static void FastAtan2_32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees)
{
	int i = 0;
	float scale = angleInDegrees ? 1 : (float)(HS_PI / 180);

#ifdef HAVE_TEGRA_OPTIMIZATION
	if (tegra::FastAtan2_32f(Y, X, angle, len, scale))
		return;
#endif

#if _USE_SSE2_
	if (_USE_SSE2_)
	{
		Hs32suf iabsmask; iabsmask.i = 0x7fffffff;
		__m128 eps = _mm_set1_ps((float)DBL_EPSILON), absmask = _mm_set1_ps(iabsmask.f);
		__m128 _90 = _mm_set1_ps(90.f), _180 = _mm_set1_ps(180.f), _360 = _mm_set1_ps(360.f);
		__m128 z = _mm_setzero_ps(), scale4 = _mm_set1_ps(scale);
		__m128 p1 = _mm_set1_ps(atan2_p1), p3 = _mm_set1_ps(atan2_p3);
		__m128 p5 = _mm_set1_ps(atan2_p5), p7 = _mm_set1_ps(atan2_p7);

		for (; i <= len - 4; i += 4)
		{
			__m128 x = _mm_loadu_ps(X + i), y = _mm_loadu_ps(Y + i);
			__m128 ax = _mm_and_ps(x, absmask), ay = _mm_and_ps(y, absmask);
			__m128 mask = _mm_cmplt_ps(ax, ay);
			__m128 tmin = _mm_min_ps(ax, ay), tmax = _mm_max_ps(ax, ay);
			__m128 c = _mm_div_ps(tmin, _mm_add_ps(tmax, eps));
			__m128 c2 = _mm_mul_ps(c, c);
			__m128 a = _mm_mul_ps(c2, p7);
			a = _mm_mul_ps(_mm_add_ps(a, p5), c2);
			a = _mm_mul_ps(_mm_add_ps(a, p3), c2);
			a = _mm_mul_ps(_mm_add_ps(a, p1), c);

			__m128 b = _mm_sub_ps(_90, a);
			a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

			b = _mm_sub_ps(_180, a);
			mask = _mm_cmplt_ps(x, z);
			a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

			b = _mm_sub_ps(_360, a);
			mask = _mm_cmplt_ps(y, z);
			a = _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(a, b), mask));

			a = _mm_mul_ps(a, scale4);
			_mm_storeu_ps(angle + i, a);
		}
	}
#endif

	for (; i < len; i++)
	{
		float x = X[i], y = Y[i];
		float ax = std::abs(x), ay = std::abs(y);
		float a, c, c2;
		if (ax >= ay)
		{
			c = ay / (ax + (float)DBL_EPSILON);
			c2 = c*c;
			a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
		}
		else
		{
			c = ax / (ay + (float)DBL_EPSILON);
			c2 = c*c;
			a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
		}
		if (x < 0)
			a = 180.f - a;
		if (y < 0)
			a = 360.f - a;
		angle[i] = (float)(a*scale);
	}
}

void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees)
{
	FastAtan2_32f(y, x, dst, n, angleInDegrees);
}
	

// magnitude
static void Magnitude_32f(const float* x, const float* y, float* mag, int len)
{
	int i = 0;

#if _USE_SSE2_
	if (_USE_SSE2_)
	{
		for (; i <= len - 8; i += 8)
		{
			__m128 x0 = _mm_loadu_ps(x + i), x1 = _mm_loadu_ps(x + i + 4);
			__m128 y0 = _mm_loadu_ps(y + i), y1 = _mm_loadu_ps(y + i + 4);
			x0 = _mm_add_ps(_mm_mul_ps(x0, x0), _mm_mul_ps(y0, y0));
			x1 = _mm_add_ps(_mm_mul_ps(x1, x1), _mm_mul_ps(y1, y1));
			x0 = _mm_sqrt_ps(x0); x1 = _mm_sqrt_ps(x1);
			_mm_storeu_ps(mag + i, x0); _mm_storeu_ps(mag + i + 4, x1);
		}
	}
#endif

	for (; i < len; i++)
	{
		float x0 = x[i], y0 = y[i];
		mag[i] = std::sqrt(x0*x0 + y0*y0);
	}
}

void magnitude(const float* x, const float* y, float* dst, int n)
{
	Magnitude_32f(x, y, dst, n);
}


/****************************************************************************************\
*                              Solving a linear system                                   *
\****************************************************************************************/
bool solve(Mat& src, Mat& src2, Mat& dst, int method)
{
	bool result = true;
	//Mat src = _src.getMat(), _src2 = _src2arg.getMat();
	int type = (src.bit_depth() >> 3);
	bool is_normal = (method & DECOMP_NORMAL) != 0;

	//CV_Assert(type == _src2.type() && (type == CV_32F || type == CV_64F));

	method &= ~DECOMP_NORMAL;
	//CV_Assert((method != DECOMP_LU && method != DECOMP_CHOLESKY) || is_normal || src.rows == src.cols);

	// check case of a single equation and small matrix
	if ((method == DECOMP_LU || method == DECOMP_CHOLESKY) && !is_normal &&
		src.height() <= 3 && src.height() == src.width() && src2.width() == 1)
	{
		dst.CreateImage(src.width(), src2.width(), src.channel(), src.bit_depth(), src.color_type());

#define bf(y) ((float*)(bdata + y*src2step))[0]
#define bd(y) ((double*)(bdata + y*src2step))[0]

		Mat::Byte* srcdata = src.GetBuffer();
		Mat::Byte* bdata = src2.GetBuffer();
		Mat::Byte* dstdata = dst.GetBuffer();
		size_t srcstep = src.width()*src.channel()*type;
		size_t src2step = src2.width()*src2.channel()*type;
		size_t dststep = dst.width()*dst.channel()*type;
		int src1h = src.height();

		if (src1h == 2)
		{
			if (type == 4) //float
			{
				double d = det2(Sf);
				if (d != 0.)
				{
					double t;
					d = 1. / d;
					t = (float)(((double)bf(0)*Sf(1, 1) - (double)bf(1)*Sf(0, 1))*d);
					Df(1, 0) = (float)(((double)bf(1)*Sf(0, 0) - (double)bf(0)*Sf(1, 0))*d);
					Df(0, 0) = (float)t;
				}
				else
					result = false;
			}
			else
			{
				double d = det2(Sd);
				if (d != 0.)
				{
					double t;
					d = 1. / d;
					t = (bd(0)*Sd(1, 1) - bd(1)*Sd(0, 1))*d;
					Dd(1, 0) = (bd(1)*Sd(0, 0) - bd(0)*Sd(1, 0))*d;
					Dd(0, 0) = t;
				}
				else
					result = false;
			}
		}
		else if (src1h == 3)
		{
			if (type == 4)
			{
				double d = det3(Sf);
				if (d != 0.)
				{
					float t[3];
					d = 1. / d;

					t[0] = (float)(d*
						(bf(0)*((double)Sf(1, 1)*Sf(2, 2) - (double)Sf(1, 2)*Sf(2, 1)) -
						Sf(0, 1)*((double)bf(1)*Sf(2, 2) - (double)Sf(1, 2)*bf(2)) +
						Sf(0, 2)*((double)bf(1)*Sf(2, 1) - (double)Sf(1, 1)*bf(2))));

					t[1] = (float)(d*
						(Sf(0, 0)*(double)(bf(1)*Sf(2, 2) - (double)Sf(1, 2)*bf(2)) -
						bf(0)*((double)Sf(1, 0)*Sf(2, 2) - (double)Sf(1, 2)*Sf(2, 0)) +
						Sf(0, 2)*((double)Sf(1, 0)*bf(2) - (double)bf(1)*Sf(2, 0))));

					t[2] = (float)(d*
						(Sf(0, 0)*((double)Sf(1, 1)*bf(2) - (double)bf(1)*Sf(2, 1)) -
						Sf(0, 1)*((double)Sf(1, 0)*bf(2) - (double)bf(1)*Sf(2, 0)) +
						bf(0)*((double)Sf(1, 0)*Sf(2, 1) - (double)Sf(1, 1)*Sf(2, 0))));

					Df(0, 0) = t[0];
					Df(1, 0) = t[1];
					Df(2, 0) = t[2];
				}
				else
					result = false;
			}
			else
			{
				double d = det3(Sd);
				if (d != 0.)
				{
					double t[9];

					d = 1. / d;

					t[0] = ((Sd(1, 1) * Sd(2, 2) - Sd(1, 2) * Sd(2, 1))*bd(0) +
						(Sd(0, 2) * Sd(2, 1) - Sd(0, 1) * Sd(2, 2))*bd(1) +
						(Sd(0, 1) * Sd(1, 2) - Sd(0, 2) * Sd(1, 1))*bd(2))*d;

					t[1] = ((Sd(1, 2) * Sd(2, 0) - Sd(1, 0) * Sd(2, 2))*bd(0) +
						(Sd(0, 0) * Sd(2, 2) - Sd(0, 2) * Sd(2, 0))*bd(1) +
						(Sd(0, 2) * Sd(1, 0) - Sd(0, 0) * Sd(1, 2))*bd(2))*d;

					t[2] = ((Sd(1, 0) * Sd(2, 1) - Sd(1, 1) * Sd(2, 0))*bd(0) +
						(Sd(0, 1) * Sd(2, 0) - Sd(0, 0) * Sd(2, 1))*bd(1) +
						(Sd(0, 0) * Sd(1, 1) - Sd(0, 1) * Sd(1, 0))*bd(2))*d;

					Dd(0, 0) = t[0];
					Dd(1, 0) = t[1];
					Dd(2, 0) = t[2];
				}
				else
					result = false;
			}
		}
		else
		{
			assert(src.height() == 1);

			if (type == 4)
			{
				double d = Sf(0, 0);
				if (d != 0.)
					Df(0, 0) = (float)(bf(0) / d);
				else
					result = false;
			}
			else
			{
				double d = Sd(0, 0);
				if (d != 0.)
					Dd(0, 0) = (bd(0) / d);
				else
					result = false;
			}
		}
		return result;
	}

	//if (method == DECOMP_QR)
	//	method = DECOMP_SVD;

	//int m = src.height(), m_ = m, n = src.width(), nb = src2.width();
	//size_t esz = CV_ELEM_SIZE(type), bufsize = 0;
	//size_t vstep = alignSize(n*esz, 16);
	//size_t astep = method == DECOMP_SVD && !is_normal ? alignSize(m*esz, 16) : vstep;
	//AutoBuffer<uchar> buffer;

	//Mat src2 = _src2;
	//_dst.create(src.cols, src2.cols, src.type());
	//Mat dst = _dst.getMat();

	//if (m < n)
	//	return false;
	//	//CV_Error(CV_StsBadArg, "The function can not solve under-determined linear systems");

	//if (m == n)
	//	is_normal = false;
	//else if (is_normal)
	//{
	//	m_ = n;
	//	if (method == DECOMP_SVD)
	//		method = DECOMP_EIG;
	//}

	//size_t asize = astep*(method == DECOMP_SVD || is_normal ? n : m);
	//bufsize += asize + 32;

	//if (is_normal)
	//	bufsize += n*nb*esz;

	//if (method == DECOMP_SVD || method == DECOMP_EIG)
	//	bufsize += n * 5 * esz + n*vstep + nb*sizeof(double) + 32;

	//buffer.allocate(bufsize);
	//uchar* ptr = alignPtr((uchar*)buffer, 16);

	//Mat a(m_, n, type, ptr, astep);

	//if (is_normal)
	//	mulTransposed(src, a, true);
	//else if (method != DECOMP_SVD)
	//	src.copyTo(a);
	//else
	//{
	//	a = Mat(n, m_, type, ptr, astep);
	//	transpose(src, a);
	//}
	//ptr += asize;

	//if (!is_normal)
	//{
	//	if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
	//		src2.copyTo(dst);
	//}
	//else
	//{
	//	// a'*b
	//	if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
	//		gemm(src, src2, 1, Mat(), 0, dst, GEMM_1_T);
	//	else
	//	{
	//		Mat tmp(n, nb, type, ptr);
	//		ptr += n*nb*esz;
	//		gemm(src, src2, 1, Mat(), 0, tmp, GEMM_1_T);
	//		src2 = tmp;
	//	}
	//}

	//if (method == DECOMP_LU)
	//{
	//	if (type == CV_32F)
	//		result = LU(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb) != 0;
	//	else
	//		result = LU(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb) != 0;
	//}
	//else if (method == DECOMP_CHOLESKY)
	//{
	//	if (type == CV_32F)
	//		result = Cholesky(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb);
	//	else
	//		result = Cholesky(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb);
	//}
	//else
	//{
	//	ptr = alignPtr(ptr, 16);
	//	Mat v(n, n, type, ptr, vstep), w(n, 1, type, ptr + vstep*n), u;
	//	ptr += n*(vstep + esz);

	//	if (method == DECOMP_EIG)
	//	{
	//		if (type == CV_32F)
	//			Jacobi(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, n, ptr);
	//		else
	//			Jacobi(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, n, ptr);
	//		u = v;
	//	}
	//	else
	//	{
	//		if (type == CV_32F)
	//			JacobiSVD(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, m_, n);
	//		else
	//			JacobiSVD(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, m_, n);
	//		u = a;
	//	}

	//	if (type == CV_32F)
	//	{
	//		SVBkSb(m_, n, w.ptr<float>(), 0, u.ptr<float>(), u.step, true,
	//			v.ptr<float>(), v.step, true, src2.ptr<float>(),
	//			src2.step, nb, dst.ptr<float>(), dst.step, ptr);
	//	}
	//	else
	//	{
	//		SVBkSb(m_, n, w.ptr<double>(), 0, u.ptr<double>(), u.step, true,
	//			v.ptr<double>(), v.step, true, src2.ptr<double>(),
	//			src2.step, nb, dst.ptr<double>(), dst.step, ptr);
	//	}
	//	result = true;
	//}

	//if (!result)
	//	dst = Scalar(0);

	//return result;
}


int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n)
{
	return LUImpl(A, astep, m, b, bstep, n);
}


int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n)
{
	return LUImpl(A, astep, m, b, bstep, n);
}

bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n)
{
	return CholImpl(A, astep, m, b, bstep, n);
}

bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n)
{
	return CholImpl(A, astep, m, b, bstep, n);
}

}
}