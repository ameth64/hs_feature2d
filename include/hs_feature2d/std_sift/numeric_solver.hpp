//#pragma once
#ifndef _HS_FEATURE2D_STD_SIFT_NUMERIC_SOLVER_HPP_
#define _HS_FEATURE2D_STD_SIFT_NUMERIC_SOLVER_HPP_

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_feature2d/std_sift/base_type.hpp"

#if (defined _M_X64 && defined _MSC_VER && _MSC_VER >= 1400) || (__GNUC__ >= 4 && defined __x86_64__)
#  if defined WIN32
#    include <intrin.h>
#  endif
#  if defined __SSE2__ || !defined __GNUC__
#    include <emmintrin.h>
#  endif
#endif


namespace hs
{
namespace feature2d
{

typedef void(*MathFunc)(const void* src, void* dst, int len);

// matrix decomposition types
enum { DECOMP_LU=0, DECOMP_SVD=1, DECOMP_EIG=2, DECOMP_CHOLESKY=3, DECOMP_QR=4, DECOMP_NORMAL=16 };

#define Sf( y, x ) ((float*)(srcdata + y*srcstep))[x]
#define Sd( y, x ) ((double*)(srcdata + y*srcstep))[x]
#define Df( y, x ) ((float*)(dstdata + y*dststep))[x]
#define Dd( y, x ) ((double*)(dstdata + y*dststep))[x]

#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))


inline int sseRound(double v)
{
#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__)
	__m128d t = _mm_set_sd(v);
	return _mm_cvtsd_si32(t);
#endif
}

inline int sseFloor(double value)
{
#if defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__ && !defined __APPLE__)
	__m128d t = _mm_set_sd(value);
	int i = _mm_cvtsd_si32(t);
	return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t, i)));
#elif defined __GNUC__
	int i = (int)value;
	return i - (i > value);
#else
	int i = cvRound(value);
	float diff = (float)(value - i);
	return i - (diff < 0);
#endif
}

inline int sseCeil(double value)
{
#if defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__&& !defined __APPLE__)
	__m128d t = _mm_set_sd(value);
	int i = _mm_cvtsd_si32(t);
	return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
#elif defined __GNUC__
	int i = (int)value;
	return i + (i < value);
#else
	int i = cvRound(value);
	float diff = (float)(i - value);
	return i + (diff < 0);
#endif
}


/************************************************************************/
/* 特化后的traits                                                         */
/************************************************************************/
template<> inline uchar saturate_cast<uchar>(schar v)
{
	return (uchar)std::max((int)v, (int)0);
}
template<> inline uchar saturate_cast<uchar>(ushort v)
{
	return (uchar)std::min((unsigned)v, (unsigned)UCHAR_MAX);
}
template<> inline uchar saturate_cast<uchar>(int v)
{
	return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}
template<> inline uchar saturate_cast<uchar>(short v)
{
	return saturate_cast<uchar>((int)v);
}
template<> inline uchar saturate_cast<uchar>(unsigned v)
{
	return (uchar)std::min(v, (unsigned)UCHAR_MAX);
}
template<> inline uchar saturate_cast<uchar>(float v)
{
	int iv = sseRound(v); return saturate_cast<uchar>(iv);
}
template<> inline uchar saturate_cast<uchar>(double v)
{
	int iv = sseRound(v); return saturate_cast<uchar>(iv);
}


//exp
static const double exp_prescale = 1.4426950408889634073599246810019 * (1 << EXPTAB_SCALE);
static const double exp_postscale = 1. / (1 << EXPTAB_SCALE);
static const double exp_max_val = 3000.*(1 << EXPTAB_SCALE); // log10(DBL_MAX) < 3000


static void Exp_32f(const float *_x, float *y, int n);

void exp(const float* src, float* dst, int n);

//atan
static const float atan2_p1 = 0.9997878412794807f*(float)(180 / HS_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180 / HS_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180 / HS_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180 / HS_PI);

static void FastAtan2_32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees = true);

void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);


// magnitude
static void Magnitude_32f(const float* x, const float* y, float* mag, int len);

void magnitude(const float* x, const float* y, float* dst, int n);



/****************************************************************************************\
*                              Solving a linear system                                   *
\****************************************************************************************/
bool solve(Mat& src, Mat& src2, Mat& dst, int method);

/****************************************************************************************\
*                     LU & Cholesky implementation for small matrices                    *
\****************************************************************************************/
template<typename _Tp> static inline int
LUImpl(_Tp* A, size_t astep, int m, _Tp* b, size_t bstep, int n)
{
	int i, j, k, p = 1;
	astep /= sizeof(A[0]);
	bstep /= sizeof(b[0]);

	for (i = 0; i < m; i++)
	{
		k = i;

		for (j = i + 1; j < m; j++)
			if (std::abs(A[j*astep + i]) > std::abs(A[k*astep + i]))
				k = j;

		if (std::abs(A[k*astep + i]) < std::numeric_limits<_Tp>::epsilon())
			return 0;

		if (k != i)
		{
			for (j = i; j < m; j++)
				std::swap(A[i*astep + j], A[k*astep + j]);
			if (b)
				for (j = 0; j < n; j++)
					std::swap(b[i*bstep + j], b[k*bstep + j]);
			p = -p;
		}

		_Tp d = -1 / A[i*astep + i];

		for (j = i + 1; j < m; j++)
		{
			_Tp alpha = A[j*astep + i] * d;

			for (k = i + 1; k < m; k++)
				A[j*astep + k] += alpha*A[i*astep + k];

			if (b)
				for (k = 0; k < n; k++)
					b[j*bstep + k] += alpha*b[i*bstep + k];
		}

		A[i*astep + i] = -d;
	}

	if (b)
	{
		for (i = m - 1; i >= 0; i--)
			for (j = 0; j < n; j++)
			{
			_Tp s = b[i*bstep + j];
			for (k = i + 1; k < m; k++)
				s -= A[i*astep + k] * b[k*bstep + j];
			b[i*bstep + j] = s*A[i*astep + i];
			}
	}

	return p;
}


int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);


int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);




template<typename _Tp> static inline bool
CholImpl(_Tp* A, size_t astep, int m, _Tp* b, size_t bstep, int n)
{
	_Tp* L = A;
	int i, j, k;
	double s;
	astep /= sizeof(A[0]);
	bstep /= sizeof(b[0]);

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < i; j++)
		{
			s = A[i*astep + j];
			for (k = 0; k < j; k++)
				s -= L[i*astep + k] * L[j*astep + k];
			L[i*astep + j] = (_Tp)(s*L[j*astep + j]);
		}
		s = A[i*astep + i];
		for (k = 0; k < j; k++)
		{
			double t = L[i*astep + k];
			s -= t*t;
		}
		if (s < std::numeric_limits<_Tp>::epsilon())
			return false;
		L[i*astep + i] = (_Tp)(1. / std::sqrt(s));
	}

	if (!b)
		return true;

	// LLt x = b
	// 1: L y = b
	// 2. Lt x = y

	/*
	[ L00             ]  y0   b0
	[ L10 L11         ]  y1 = b1
	[ L20 L21 L22     ]  y2   b2
	[ L30 L31 L32 L33 ]  y3   b3

	[ L00 L10 L20 L30 ]  x0   y0
	[     L11 L21 L31 ]  x1 = y1
	[         L22 L32 ]  x2   y2
	[             L33 ]  x3   y3
	*/

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			s = b[i*bstep + j];
			for (k = 0; k < i; k++)
				s -= L[i*astep + k] * b[k*bstep + j];
			b[i*bstep + j] = (_Tp)(s*L[i*astep + i]);
		}
	}

	for (i = m - 1; i >= 0; i--)
	{
		for (j = 0; j < n; j++)
		{
			s = b[i*bstep + j];
			for (k = m - 1; k > i; k--)
				s -= L[k*astep + i] * b[k*bstep + j];
			b[i*bstep + j] = (_Tp)(s*L[i*astep + i]);
		}
	}

	return true;
}

// call entry
bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);




}
}

#endif //_HS_FEATURE2D_STD_SIFT_NUMERIC_SOLVER_HPP_


