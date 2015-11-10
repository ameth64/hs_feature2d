#ifndef _HS_FEATURE2D_STD_SIFT_MATRIX_HPP_
#define _HS_FEATURE2D_STD_SIFT_MATRIX_HPP_

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "hs_feature2d/config/hs_config.hpp"
#include "hs_image_io/whole_io/image_data.hpp"
#include "hs_feature2d/std_sift/base_type.hpp"
#include "hs_feature2d/std_sift/numeric_solver.hpp"


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

template<typename _Tp, int cn> class Vec;
// the matrix template class, represents a m*n matrix 
template<typename _Tp, int m, int n> class Matx
{
public:
    typedef _Tp value_type;
    typedef Matx<_Tp, (m < n ? m : n), 1> diag_type;
    typedef Matx<_Tp, m, n> mat_type;
    enum { depth = DataDepth<_Tp>::value, rows = m, cols = n, channels = rows*cols,
           type = T_MAKETYPE(depth, channels) };

    //! default constructor
    Matx()
	{
		for (int i = 0; i < channels; i++) val[i] = _Tp(0);
	};

    Matx(_Tp v0)
	{
		val[0] = v0;
		for (int i = 1; i < channels; i++) val[i] = _Tp(0);
	}; //!< 1x1 matrix
    Matx(_Tp v0, _Tp v1)
	{
		val[0] = v0; val[1] = v1;
		for (int i = 2; i < channels; i++) val[i] = _Tp(0);
	}; //!< 1x2 or 2x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2)
	{
		val[0] = v0; val[1] = v1; val[2] = v2;
		for (int i = 3; i < channels; i++) val[i] = _Tp(0);
	}; //!< 1x3 or 3x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
	{
		val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
		for (int i = 4; i < channels; i++) val[i] = _Tp(0);
	}; //!< 1x4, 2x2 or 4x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4)
	{
		val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3; val[4] = v4;
		for (int i = 5; i < channels; i++) val[i] = _Tp(0);
	}; //!< 1x5 or 5x1 matrix

	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8)
	{
		val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
		val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
		val[8] = v8;
		for (int i = 9; i < channels; i++) val[i] = _Tp(0);
	}


	//!< initialize from a plain array
	explicit Matx(const _Tp* values)
	{
		for (int i = 0; i < channels; i++) val[i] = values[i];
	};

	//! Copy data to a Mat object
	bool CopyTo(Mat& dst) const
	{
		bool res = (0 == dst.CreateImage(n, m, 1, sizeof(_Tp)*Mat::ByteDepth, Mat::IMAGE_GRAYSCALE));
		_Tp* pdst = dst.GetBufferT<_Tp>();
		memcpy(pdst, this->val, m*n*sizeof(_Tp));
		return res;
	}

	// set all the elements in matx
    static Matx all(_Tp alpha)
	{
		Matx<_Tp, m, n> M;
		for (int i = 0; i < m*n; i++) M.val[i] = alpha;
		return M;
	};

    static Matx zeros(){ return all(0); };

	static Matx ones(){ return all(1); };

    static Matx eye()
	{
		Matx<_Tp, m, n> M;
		for (int i = 0; i < MIN(m, n); i++)
			M(i, i) = 1;
		return M;
	};

    static Matx diag(const diag_type& d)
	{
		Matx<_Tp, m, n> M;
		for (int i = 0; i < MIN(m, n); i++)
			M(i, i) = d(i, 0);
		return M;
	};

    static Matx randu(_Tp a, _Tp b)
	{
		Matx<_Tp, m, n> M;
		Mat matM(M, false);
		//cv::randu(matM, Scalar(a), Scalar(b));
		return M;
	};

    static Matx randn(_Tp a, _Tp b)
	{
		Matx<_Tp, m, n> M;
		Mat matM(M, false);
		//cv::randn(matM, Scalar(a), Scalar(b));
		return M;
	};

    //! dot product computed with the default precision
	_Tp dot(const Matx<_Tp, m, n>& M) const
	{
		_Tp s = 0;
		for (int i = 0; i < channels; i++) s += val[i] * M.val[i];
		return s;
	}

    //! dot product computed in double-precision arithmetics
    double ddot(const Matx<_Tp, m, n>& v) const
	{
		double s = 0;
		for (int i = 0; i < m*n; i++) s += (double)val[i] * M.val[i];
		return s;
	};

    //! change the matrix shape
    template<int m1, int n1> Matx<_Tp, m1, n1> reshape() const;

    //! extract part of the matrix
    template<int m1, int n1> Matx<_Tp, m1, n1> get_minor(int i, int j) const;

    //! extract the matrix row
    Matx<_Tp, 1, n> row(int i) const;

    //! extract the matrix column
    Matx<_Tp, m, 1> col(int i) const;

    //! extract the matrix diagonal
    diag_type diag() const;

    //! transpose the matrix
    Matx<_Tp, n, m> t() const;

    //! invert matrix the matrix
    Matx<_Tp, n, m> inv(int method=DECOMP_LU) const;

    //! solve linear system
    template<int l>
	Matx<_Tp, n, l> solve(const Matx<_Tp, m, l>& rhs, int method = DECOMP_LU) const
	{
		Matx<_Tp, n, l> x;
		bool ok;
		if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
			ok = Matx_FastSolveOp<_Tp, m, l>()(*this, rhs, x, method);
		else
		{
			Mat A, B, X;
			this->CopyTo(A);
			rhs.CopyTo(B);
			x.CopyTo(X);
			ok = hs::feature2d::solve(A, B, X, method);
		}

		return ok ? x : Matx<_Tp, n, l>::zeros();
	};
	
    Vec<_Tp, n> solve(const Vec<_Tp, m>& rhs, int method) const
	{
		Matx<_Tp, n, 1> x = solve(reinterpret_cast<const Matx<_Tp, m, 1>&>(rhs), method);
		return reinterpret_cast<Vec<_Tp, n>&>(x);
	};

    //! multiply two matrices element-wise
    Matx<_Tp, m, n> mul(const Matx<_Tp, m, n>& a) const;

	//! convertion to another data type
	template<typename T2> operator Matx<T2, m, n>() const
	{
		Matx<T2, m, n> M;
		for (int i = 0; i < m*n; i++) M.val[i] = saturate_cast<T2>(val[i]);
		return M;
	};

    //! element access
    const _Tp& operator ()(int i, int j) const
	{
		return val[i*n + j];
	};
    _Tp& operator ()(int i, int j)
	{
		return val[i*n + j];
	};

    //! 1D element access
    const _Tp& operator ()(int i) const
	{
		return val[i];
	};
    _Tp& operator ()(int i)
	{
		return val[i];
	};

    //Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_AddOp);
    //Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_SubOp);
    //template<typename _T2> Matx(const Matx<_Tp, m, n>& a, _T2 alpha, Matx_ScaleOp);
    //Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_MulOp);
    //template<int l> Matx(const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b, Matx_MatMulOp);
    //Matx(const Matx<_Tp, n, m>& a, Matx_TOp);

    _Tp val[m*n]; //< matrix elements
};

typedef Matx<float, 1, 2> Matx12f;
typedef Matx<double, 1, 2> Matx12d;
typedef Matx<float, 1, 3> Matx13f;
typedef Matx<double, 1, 3> Matx13d;
typedef Matx<float, 1, 4> Matx14f;
typedef Matx<double, 1, 4> Matx14d;
typedef Matx<float, 1, 6> Matx16f;
typedef Matx<double, 1, 6> Matx16d;

typedef Matx<float, 2, 1> Matx21f;
typedef Matx<double, 2, 1> Matx21d;
typedef Matx<float, 3, 1> Matx31f;
typedef Matx<double, 3, 1> Matx31d;
typedef Matx<float, 4, 1> Matx41f;
typedef Matx<double, 4, 1> Matx41d;
typedef Matx<float, 6, 1> Matx61f;
typedef Matx<double, 6, 1> Matx61d;

typedef Matx<float, 2, 2> Matx22f;
typedef Matx<double, 2, 2> Matx22d;
typedef Matx<float, 2, 3> Matx23f;
typedef Matx<double, 2, 3> Matx23d;
typedef Matx<float, 3, 2> Matx32f;
typedef Matx<double, 3, 2> Matx32d;

typedef Matx<float, 3, 3> Matx33f;
typedef Matx<double, 3, 3> Matx33d;

typedef Matx<float, 3, 4> Matx34f;
typedef Matx<double, 3, 4> Matx34d;
typedef Matx<float, 4, 3> Matx43f;
typedef Matx<double, 4, 3> Matx43d;

typedef Matx<float, 4, 4> Matx44f;
typedef Matx<double, 4, 4> Matx44d;
typedef Matx<float, 6, 6> Matx66f;
typedef Matx<double, 6, 6> Matx66d;


template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1>
{
public:
    typedef _Tp value_type;
    enum { depth = DataDepth<_Tp>::value, channels = cn, type = T_MAKETYPE(depth, channels) };

    //! default constructor
	Vec(){};

	Vec(_Tp v0) : Matx<_Tp, cn, 1>(v0)
	{}; //!< 1-element vector constructor
	Vec(_Tp v0, _Tp v1) : Matx<_Tp, cn, 1>(v0, v1)
	{}; //!< 2-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2) : Matx<_Tp, cn, 1>(v0, v1, v2)
	{}; //!< 3-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3) : Matx<_Tp, cn, 1>(v0, v1, v2, v3)
	{}; //!< 4-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4)
	{}; //!< 5-element vector constructor

	//explicit Vec(const _Tp* values) : Matx<_Tp, cn, 1>(values){};

	Vec(const Vec<_Tp, cn>& v) : Matx<_Tp, cn, 1>(v.val) {};

	inline const _Tp& operator()(int i, int j) const
	{
		return this->val[i*n + j];
	}

    static Vec all(_Tp alpha);

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn>& v) const;

    //! conjugation (makes sense for complex numbers and quaternions)
    Vec conj() const;

    /*! cross product of the two 3D vectors.
      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec& v) const;
    //! convertion to another data type
    template<typename T2> operator Vec<T2, cn>() const;
    //! conversion to 4-element CvScalar.
    //operator CvScalar() const;

    /*! element access */
    const _Tp& operator [](int i) const
	{
		return this->val[i];
	};
    _Tp& operator[](int i)
	{
		return this->val[i];
	};
    const _Tp& operator ()(int i) const
	{
		return this->val[i];
	};
    _Tp& operator ()(int i)
	{
		return this->val[i];
	};

    //Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_AddOp);
    //Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_SubOp);
    //template<typename _T2> Vec(const Matx<_Tp, cn, 1>& a, _T2 alpha, Matx_ScaleOp);
};

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

//template<> inline Vec<float, 3> Vec<float, 3>::cross(const Vec<float, 3>& v) const
//{
//	return Vec<float, 3>(val[1] * v.val[2] - val[2] * v.val[1],
//		val[2] * v.val[0] - val[0] * v.val[2],
//		val[0] * v.val[1] - val[1] * v.val[0]);
//}

template<typename _Tp, int m> static inline
double determinant(const Matx<_Tp, m, m>& a)
{
	return Matx_DetOp<_Tp, m>()(a);
}

// 2-3维矩阵特征值
template<typename _Tp, int m> struct Matx_DetOp
{
	double operator ()(const Matx<_Tp, m, m>& a) const
	{
		Matx<_Tp, m, m> temp = a;
		double p = LU(temp.val, m*sizeof(_Tp), m, 0, 0, 0);
		if (p == 0)
			return p;
		for (int i = 0; i < m; i++)
			p *= temp(i, i);
		return 1. / p;
	}
};
template<typename _Tp> struct Matx_DetOp<_Tp, 1>
{
    double operator ()(const Matx<_Tp, 1, 1>& a) const
    {
        return a(0,0);
    }
};


template<typename _Tp> struct Matx_DetOp<_Tp, 2>
{
    double operator ()(const Matx<_Tp, 2, 2>& a) const
    {
        return a(0,0)*a(1,1) - a(0,1)*a(1,0);
    }
};


template<typename _Tp> struct Matx_DetOp<_Tp, 3>
{
    double operator ()(const Matx<_Tp, 3, 3>& a) const
    {
        return a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2)) -
            a(0,1)*(a(1,0)*a(2,2) - a(2,0)*a(1,2)) +
            a(0,2)*(a(1,0)*a(2,1) - a(2,0)*a(1,1));
    }
};

// 2-3维矩阵求解对象
template<typename _Tp, int m, int n> struct Matx_FastSolveOp
{
	bool operator()(const Matx<_Tp, m, m>& a, const Matx<_Tp, m, n>& b,
		Matx<_Tp, m, n>& x, int method) const
	{
		Matx<_Tp, m, m> temp = a;
		x = b;
		if (method == DECOMP_CHOLESKY)
			return Cholesky(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n);

		return LU(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n) != 0;
	}
};

template<typename _Tp> 
struct Matx_FastSolveOp < _Tp, 2, 1 >
{
	bool operator()(const Matx<_Tp, 2, 2>& a, const Matx<_Tp, 2, 1>& b,
		Matx<_Tp, 2, 1>& x, int) const
	{
		_Tp d = determinant(a);
		if (d == 0)
			return false;
		d = 1 / d;
		x(0) = (b(0)*a(1, 1) - b(1)*a(0, 1))*d;
		x(1) = (b(1)*a(0, 0) - b(0)*a(1, 0))*d;
		return true;
	}
};

template<typename _Tp> struct Matx_FastSolveOp < _Tp, 3, 1 >
{
	bool operator()(const Matx<_Tp, 3, 3>& a, const Matx<_Tp, 3, 1>& b,
		Matx<_Tp, 3, 1>& x, int) const
	{
		_Tp d = (_Tp)determinant(a);
		if (d == 0)
			return false;
		d = 1 / d;
		x(0) = d*(b(0)*(a(1, 1)*a(2, 2) - a(1, 2)*a(2, 1)) -
			a(0, 1)*(b(1)*a(2, 2) - a(1, 2)*b(2)) +
			a(0, 2)*(b(1)*a(2, 1) - a(1, 1)*b(2)));

		x(1) = d*(a(0, 0)*(b(1)*a(2, 2) - a(1, 2)*b(2)) -
			b(0)*(a(1, 0)*a(2, 2) - a(1, 2)*a(2, 0)) +
			a(0, 2)*(a(1, 0)*b(2) - b(1)*a(2, 0)));

		x(2) = d*(a(0, 0)*(a(1, 1)*b(2) - b(1)*a(2, 1)) -
			a(0, 1)*(a(1, 0)*b(2) - b(1)*a(2, 0)) +
			b(0)*(a(1, 0)*a(2, 1) - a(1, 1)*a(2, 0)));
		return true;
	}
};



}
}

#endif //_HS_FEATURE2D_STD_SIFT_MATRIX_HPP_


