#ifndef _HS_FEATURE2D_STD_SIFT_BASE_TYPE_HPP_
#define _HS_FEATURE2D_STD_SIFT_BASE_TYPE_HPP_

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"

#if defined __ICL
#  define HS_ICC   __ICL
#elif defined __ICC
#  define HS_ICC   __ICC
#elif defined __ECL
#  define HS_ICC   __ECL
#elif defined __ECC
#  define HS_ICC   __ECC
#elif defined __INTEL_COMPILER
#  define HS_ICC   __INTEL_COMPILER
#endif

#if defined HS_ICC && !defined HS_ENABLE_UNROLLED
#  define HS_ENABLE_UNROLLED 0
#else
#  define HS_ENABLE_UNROLLED 1
#endif

#if (defined _M_X64 && defined _MSC_VER && _MSC_VER >= 1400) || (__GNUC__ >= 4 && defined __x86_64__)
#  if defined WIN32
#    include <intrin.h>
#  endif
#  if defined __SSE2__ || !defined __GNUC__
#    include <emmintrin.h>
#  endif
#endif

//强制inline定义
#ifdef _MSC_VER // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

//地址对齐定义
#ifdef __GNUC__
#  define _DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
#  define _DECL_ALIGNED(x) __declspec(align(x))
#else
#  define _DECL_ALIGNED(x)
#endif

#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__)
#  define _USE_SSE2_ 1
#endif

#define HS_MAX_BIT_DEPTH 8
#define HS_BYTE_ALIGN 16

#define FLT_EPSILON     1.192092896e-07F        /* smallest such that 1.0+FLT_EPSILON != 1.0 */
#define FLT_MAX         3.402823466e+38F        /* max value */
#define FLT_MIN         1.175494351e-38F        /* min positive value */

#define DBL_EPSILON     2.2204460492503131e-016 /* smallest such that 1.0+DBL_EPSILON != 1.0 */
#define DBL_MAX         1.7976931348623158e+308 /* max value */
#define DBL_MIN         2.2250738585072014e-308 /* min positive value */

#define HS_PI   3.1415926535897932384626433832795
#define HS_LOG2 0.69314718055994530941723212145818

#define EXPTAB_SCALE 6
#define EXPTAB_MASK  ((1 << EXPTAB_SCALE) - 1)
#define EXPPOLY_32F_A0 .9670371139572337719125840413672004409288e-2
static const double expTab[] = {
	1.0 * EXPPOLY_32F_A0,
	1.0108892860517004600204097905619 * EXPPOLY_32F_A0,
	1.0218971486541166782344801347833 * EXPPOLY_32F_A0,
	1.0330248790212284225001082839705 * EXPPOLY_32F_A0,
	1.0442737824274138403219664787399 * EXPPOLY_32F_A0,
	1.0556451783605571588083413251529 * EXPPOLY_32F_A0,
	1.0671404006768236181695211209928 * EXPPOLY_32F_A0,
	1.0787607977571197937406800374385 * EXPPOLY_32F_A0,
	1.0905077326652576592070106557607 * EXPPOLY_32F_A0,
	1.1023825833078409435564142094256 * EXPPOLY_32F_A0,
	1.1143867425958925363088129569196 * EXPPOLY_32F_A0,
	1.126521618608241899794798643787 * EXPPOLY_32F_A0,
	1.1387886347566916537038302838415 * EXPPOLY_32F_A0,
	1.151189229952982705817759635202 * EXPPOLY_32F_A0,
	1.1637248587775775138135735990922 * EXPPOLY_32F_A0,
	1.1763969916502812762846457284838 * EXPPOLY_32F_A0,
	1.1892071150027210667174999705605 * EXPPOLY_32F_A0,
	1.2021567314527031420963969574978 * EXPPOLY_32F_A0,
	1.2152473599804688781165202513388 * EXPPOLY_32F_A0,
	1.2284805361068700056940089577928 * EXPPOLY_32F_A0,
	1.2418578120734840485936774687266 * EXPPOLY_32F_A0,
	1.2553807570246910895793906574423 * EXPPOLY_32F_A0,
	1.2690509571917332225544190810323 * EXPPOLY_32F_A0,
	1.2828700160787782807266697810215 * EXPPOLY_32F_A0,
	1.2968395546510096659337541177925 * EXPPOLY_32F_A0,
	1.3109612115247643419229917863308 * EXPPOLY_32F_A0,
	1.3252366431597412946295370954987 * EXPPOLY_32F_A0,
	1.3396675240533030053600306697244 * EXPPOLY_32F_A0,
	1.3542555469368927282980147401407 * EXPPOLY_32F_A0,
	1.3690024229745906119296011329822 * EXPPOLY_32F_A0,
	1.3839098819638319548726595272652 * EXPPOLY_32F_A0,
	1.3989796725383111402095281367152 * EXPPOLY_32F_A0,
	1.4142135623730950488016887242097 * EXPPOLY_32F_A0,
	1.4296133383919700112350657782751 * EXPPOLY_32F_A0,
	1.4451808069770466200370062414717 * EXPPOLY_32F_A0,
	1.4609177941806469886513028903106 * EXPPOLY_32F_A0,
	1.476826145939499311386907480374 * EXPPOLY_32F_A0,
	1.4929077282912648492006435314867 * EXPPOLY_32F_A0,
	1.5091644275934227397660195510332 * EXPPOLY_32F_A0,
	1.5255981507445383068512536895169 * EXPPOLY_32F_A0,
	1.5422108254079408236122918620907 * EXPPOLY_32F_A0,
	1.5590044002378369670337280894749 * EXPPOLY_32F_A0,
	1.5759808451078864864552701601819 * EXPPOLY_32F_A0,
	1.5931421513422668979372486431191 * EXPPOLY_32F_A0,
	1.6104903319492543081795206673574 * EXPPOLY_32F_A0,
	1.628027421857347766848218522014 * EXPPOLY_32F_A0,
	1.6457554781539648445187567247258 * EXPPOLY_32F_A0,
	1.6636765803267364350463364569764 * EXPPOLY_32F_A0,
	1.6817928305074290860622509524664 * EXPPOLY_32F_A0,
	1.7001063537185234695013625734975 * EXPPOLY_32F_A0,
	1.7186192981224779156293443764563 * EXPPOLY_32F_A0,
	1.7373338352737062489942020818722 * EXPPOLY_32F_A0,
	1.7562521603732994831121606193753 * EXPPOLY_32F_A0,
	1.7753764925265212525505592001993 * EXPPOLY_32F_A0,
	1.7947090750031071864277032421278 * EXPPOLY_32F_A0,
	1.8142521755003987562498346003623 * EXPPOLY_32F_A0,
	1.8340080864093424634870831895883 * EXPPOLY_32F_A0,
	1.8539791250833855683924530703377 * EXPPOLY_32F_A0,
	1.8741676341102999013299989499544 * EXPPOLY_32F_A0,
	1.8945759815869656413402186534269 * EXPPOLY_32F_A0,
	1.9152065613971472938726112702958 * EXPPOLY_32F_A0,
	1.9360617934922944505980559045667 * EXPPOLY_32F_A0,
	1.9571441241754002690183222516269 * EXPPOLY_32F_A0,
	1.9784560263879509682582499181312 * EXPPOLY_32F_A0,
};

#define T_CN_MAX     512
#define T_CN_SHIFT   3
#define T_DEPTH_MAX  (1 << T_CN_SHIFT)

#define T_8U   0
#define T_8S   1
#define T_16U  2
#define T_16S  3
#define T_32S  4
#define T_32F  5
#define T_64F  6
#define T_USRTYPE1 7

#define T_MAT_DEPTH_MASK       (T_DEPTH_MAX - 1)
#define T_MAT_DEPTH(flags)     ((flags) & T_MAT_DEPTH_MASK)

#define T_MAKETYPE(depth,cn) (T_MAT_DEPTH(depth) + (((cn)-1) << T_CN_SHIFT))
#define T_MAKE_TYPE T_MAKETYPE

namespace hs
{
namespace feature2d
{


typedef hs::imgio::whole::ImageData Mat;
//
typedef signed char schar;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

typedef union Hs32suf
{
    int i;
    unsigned u;
    float f;
}
Hs32suf;

typedef long long int64;
typedef unsigned long long uint64;
typedef union Hs64suf
{
    int64 i;
    uint64 u;
    double f;
}
Hs64suf;

//type traits
template<typename _Tp> class DataType
{
public:
    typedef _Tp         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 1,
           depth        = -1,
           channels     = 1,
           fmt          = 0,
           type = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<bool>
{
public:
    typedef bool        value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<uchar>
{
public:
    typedef uchar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<schar>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<char>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<ushort>
{
public:
    typedef ushort      value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_16U,
           channels     = 1,
           fmt          = (int)'w',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<short>
{
public:
    typedef short       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_16S,
           channels     = 1,
           fmt          = (int)'s',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<int>
{
public:
    typedef int         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_32S,
           channels     = 1,
           fmt          = (int)'i',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<float>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_32F,
           channels     = 1,
           fmt          = (int)'f',
           type         = T_MAKETYPE(depth, channels)
         };
};

template<> class DataType<double>
{
public:
    typedef double      value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = T_64F,
           channels     = 1,
           fmt          = (int)'d',
           type         = T_MAKETYPE(depth, channels)
         };
};

// data bit depth
template<typename _Tp> class DataDepth {};

template<> class DataDepth<bool> { public: enum { value = T_8U, fmt=(int)'u' }; };
template<> class DataDepth<uchar> { public: enum { value = T_8U, fmt=(int)'u' }; };
template<> class DataDepth<schar> { public: enum { value = T_8S, fmt=(int)'c' }; };
template<> class DataDepth<char> { public: enum { value = T_8S, fmt=(int)'c' }; };
template<> class DataDepth<ushort> { public: enum { value = T_16U, fmt=(int)'w' }; };
template<> class DataDepth<short> { public: enum { value = T_16S, fmt=(int)'s' }; };
template<> class DataDepth<int> { public: enum { value = T_32S, fmt=(int)'i' }; };
// this is temporary solution to support 32-bit unsigned integers
template<> class DataDepth<unsigned> { public: enum { value = T_32S, fmt=(int)'i' }; };
template<> class DataDepth<float> { public: enum { value = T_32F, fmt=(int)'f' }; };
template<> class DataDepth<double> { public: enum { value = T_64F, fmt=(int)'d' }; };
template<typename _Tp> class DataDepth<_Tp*> { public: enum { value = T_USRTYPE1, fmt=(int)'r' }; };


/************************************************************************/
/* 类型转换的traits                                                         */
/************************************************************************/
template<typename _Tp> static inline _Tp saturate_cast(uchar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(schar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(ushort v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(short v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(unsigned v) { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int v)      { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int64 v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(uint64 v)   { return _Tp(v); }


template<typename T> class Point2D_T
{
public:
	typedef T value_type;

	Point2D_T() : x(0), y(0) {};
	Point2D_T(T _x, T _y) : x(_x), y(_y) {};
	Point2D_T(const Point2D_T& pt) : x(pt.x), y(pt.y) {};

	Point2D_T& operator = (const Point2D_T& pt)
	{
		x = pt.x; y = pt.y;
		return *this;
	};

	//数据类型转换
	template<typename T2> operator Point2D_T<T2>() const
	{
		return Point_<T2>(saturate_cast<T2>(x), saturate_cast<T2>(y));
	};

	//! dot product
	//T dot(const Point2D_T& pt) const;

	//! 强制返回double类型的点乘
	//double ddot(const Point2D_T& pt) const;

	//! 强制返回double类型的叉乘
	//double cross(const Point2D_T& pt) const;

	//! 检测点是否位于给定的矩形范围内
	//bool inside(const Rect_<T>& r) const;

	T x, y; //< the point coordinates
};

//对Point对象的重载运算符
template<typename _Tp> static inline Point2D_T<_Tp>&
operator += (Point2D_T<_Tp>& a, const Point2D_T<_Tp>& b)
{
	a.x = saturate_cast<_Tp>(a.x + b.x);
	a.y = saturate_cast<_Tp>(a.y + b.y);
	return a;
}

template<typename _Tp> static inline Point2D_T<_Tp>&
operator -= (Point2D_T<_Tp>& a, const Point2D_T<_Tp>& b)
{
	a.x = saturate_cast<_Tp>(a.x - b.x);
	a.y = saturate_cast<_Tp>(a.y - b.y);
	return a;
}

template<typename _Tp> static inline Point2D_T<_Tp>&
operator *= (Point2D_T<_Tp>& a, int b)
{
	a.x = saturate_cast<_Tp>(a.x*b);
	a.y = saturate_cast<_Tp>(a.y*b);
	return a;
}

template<typename _Tp> static inline Point2D_T<_Tp>&
operator *= (Point2D_T<_Tp>& a, float b)
{
	a.x = saturate_cast<_Tp>(a.x*b);
	a.y = saturate_cast<_Tp>(a.y*b);
	return a;
}

template<typename _Tp> static inline Point2D_T<_Tp>&
operator *= (Point2D_T<_Tp>& a, double b)
{
	a.x = saturate_cast<_Tp>(a.x*b);
	a.y = saturate_cast<_Tp>(a.y*b);
	return a;
}

typedef Point2D_T<float> Point2f;
typedef Point2D_T<int> Point;


//关键点描述对象
template<typename T> class KeyPoint_T
{
public:
	//! the default constructor
	KeyPoint_T() : pt(0, 0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}
	//! the full constructor
	KeyPoint_T(Point2f _pt, float _size, float _angle = -1,
		float _response = 0, int _octave = 0, int _class_id = -1)
		: pt(_pt), size(_size), angle(_angle),
		response(_response), octave(_octave), class_id(_class_id) {}
	//! another form of the full constructor
	KeyPoint_T(float x, float y, float _size, float _angle = -1,
		float _response = 0, int _octave = 0, int _class_id = -1)
		: pt(x, y), size(_size), angle(_angle),
		response(_response), octave(_octave), class_id(_class_id) {}

	size_t hash() const;

	//! converts vector of keypoints to vector of points
	static void convert(const std::vector< KeyPoint_T >& keypoints,
		std::vector< Point2f >& points2f,
		const std::vector<int>& keypointIndexes = vector<int>());

	//! converts vector of points to the vector of keypoints, where each keypoint is assigned the same size and the same orientation
	static void convert(const std::vector< Point2f >& points2f,
		std::vector< KeyPoint_T >& keypoints,
		float size = 1, float response = 1, int octave = 0, int class_id = -1);

	//! computes overlap for pair of keypoints;
	//! overlap is a ratio between area of keypoint regions intersection and
	//! area of keypoint regions union (now keypoint region is circle)
	static float overlap(const KeyPoint_T& kp1, const KeyPoint_T& kp2);


	Point2D_T< T > pt; //!< coordinates of the keypoints
	T size; //!< diameter of the meaningful keypoint neighborhood
	T angle; //!< computed orientation of the keypoint (-1 if not applicable);
	//!< it's in [0,360) degrees and measured relative to
	//!< image coordinate system, ie in clockwise.
	T response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
	int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
	int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)
};

typedef KeyPoint_T<float> KeyPoint;


class HS_EXPORT KeyPointsFilter{
public:
	KeyPointsFilter(){}

	/*
	* Remove keypoints within borderPixels of an image edge.
	*/
	//static void runByImageBorder(std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize);
	/*
	* Remove keypoints of sizes out of range.
	*/
	static void runByKeypointSize(std::vector<KeyPoint>& keypoints, float minSize,
		float maxSize = FLT_MAX);
	/*
	* Remove keypoints from some image by mask for pixels of this image.
	*/
	static void runByPixelsMask(std::vector<KeyPoint>& keypoints, const Mat& mask);
	/*
	* Remove duplicated keypoints.
	*/
	static void removeDuplicated(std::vector<KeyPoint>& keypoints);

	/*
	* Retain the specified number of the best keypoints (according to the response)
	*/
	static void retainBest(std::vector<KeyPoint>& keypoints, int npoints);
};

struct KeypointResponseGreaterThanThreshold
{
	KeypointResponseGreaterThanThreshold(float _value) :
		value(_value)
	{
	}
	inline bool operator()(const KeyPoint& kpt) const
	{
		return kpt.response >= value;
	}
	float value;
};

struct KeypointResponseGreater
{
	inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
	{
		return kp1.response > kp2.response;
	}
};

}
}

#endif //_HS_FEATURE2D_STD_SIFT_BASE_TYPE_HPP_


