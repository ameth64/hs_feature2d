#ifndef _HS_FEATURE2D_STD_SIFT_ARRAYHELPER_H
#define _HS_FEATURE2D_STD_SIFT_ARRAYHELPER_H

#include <vector>

namespace hs
{
namespace feature2d
{

#define  F2D_PTR_ALIGN    16

//堆分配数组管理类
struct array_del{
	template< class ptr >
	inline void operator()(ptr* p)
	{
		if(p != NULL)
		{
			delete[] p;
			p = NULL;
		}
	}
};
struct single_del{
	template< class ptr >
	inline void operator()(ptr* p){
		if(p != NULL)
		{
			delete p;
			p = NULL;
		}
	}
};

//内存对齐的堆分配数组管理类
struct aligned_heap_del{
	template< class ptr >
	inline void operator()(ptr* p)
	{
		if (p != NULL)
		{
			_aligned_free(p);
			p = NULL;
		}
	}
};

template< typename T >
class HeapMgr
{
	T*	_ptr;
	array_del	_aop;
	single_del	_sop;
	size_t _size;
public:
	HeapMgr(size_t s): _ptr((s > 1)?(new T[s]):((s == 1)?(new T):NULL)), _size(s)
	{
		memset(_ptr, 0, _size*sizeof(T));
	}
	HeapMgr(T* t): _ptr(t), _size(1)
	{}
	HeapMgr(const HeapMgr& hm): _size(hm._size >= 0?(hm._size):(0)),
		_ptr(NULL)
	{
		if (_size == 1)
		{
			_ptr = new T(*(hm._ptr));
		}
		if (_size > 1)
		{
			_ptr = new T[_size];
			for (int k = 0; k < _size; k++)
			{
				_ptr[k] = hm._ptr[k];
			}
		}
	}
	HeapMgr(): _ptr(NULL), _size(0)
	{}

	HeapMgr& operator= (const HeapMgr& hm)
	{
		_size = hm._size, _ptr = NULL;
		if (_size == 1)
		{
			_ptr = new T(*(hm._ptr));
		}
		if (_size > 1)
		{
			_ptr = new T[_size];
			for (int k = 0; k < _size; k++)
			{
				_ptr[k] = hm._ptr[k];
			}
		}
		return *this;
	}

	operator T* () { return _ptr; };
	operator const T* () const { return _ptr; };

	~HeapMgr(){
		(_size > 1)?(_aop(_ptr)):(_sop(_ptr));
	}
	bool Allocate(size_t s = 1){
		if(s < 1) return false;
		(_size > 1)?(_aop(_ptr)):(_sop(_ptr));
		_size = s;
		_ptr = (s > 1)?(new T[s]):(new T);
		return (_ptr != NULL);
	}
	T* GetPtr(){return _ptr;};
	size_t GetSize(){return _size;};

	//T& operator() (size_t index)
};

//内存对齐的堆分配管理类
template< typename T >
class HeapMgrA
{
	T*	_ptr;
	aligned_heap_del	_aop;
	size_t _size;
public:
	HeapMgrA(size_t s) : _ptr((s > 0) ? ( (T*)_aligned_malloc(s*sizeof(T), F2D_PTR_ALIGN) ) : NULL), _size(s)
	{
		memset(_ptr, 0, _size*sizeof(T));
	}
	HeapMgrA(T* t) : _ptr(t), _size(1)
	{}
	HeapMgrA(const HeapMgrA& hm) : _size(hm._size >= 0 ? (hm._size) : (0)),
		_ptr(NULL)
	{
		_ptr = (T*)_aligned_malloc(_size * sizeof(T), F2D_PTR_ALIGN);
		for (int k = 0; k < _size; k++)
		{
			_ptr[k] = hm._ptr[k];
		}
	}
	HeapMgrA() : _ptr(NULL), _size(0)
	{}

	HeapMgrA& operator= (const HeapMgrA& hm)
	{
		_size = hm._size, _ptr = NULL;
		if (_size > 0)
		{
			_ptr = (T*)_aligned_malloc(_size * sizeof(T), F2D_PTR_ALIGN);
			for (int k = 0; k < _size; k++)
			{
				_ptr[k] = hm._ptr[k];
			}
		}
		return *this;
	}

	~HeapMgrA(){
		_aop(_ptr);
	}
	bool Allocate(size_t s = 1){
		if (s < 1) return false;
		_aop(_ptr);
		_size = s;
		_ptr = (T*)_aligned_malloc(_size * sizeof(T), F2D_PTR_ALIGN);
		return (_ptr != NULL);
	}
	T* GetPtr(){ return _ptr; };
	size_t GetSize(){ return _size; };

	//T& operator() (size_t index)
};


template< typename T >
class ResUnitMgr{
	T* _ptr;
public:
	ResUnitMgr(): _ptr(NULL){}

	ResUnitMgr(const T &t): _ptr(new T(t))
	{}

	ResUnitMgr(const ResUnitMgr &rm): _ptr(new T(*(rm._ptr)))
	{}

	ResUnitMgr(bool alloc): _ptr(alloc? (new T): NULL)
	{}

	~ResUnitMgr(){
		if(_ptr != NULL)
			delete _ptr;
	}

	inline const ResUnitMgr& operator= (const T& t)
	{
		if(_ptr != NULL)
			delete _ptr;
		_ptr = new T(t);
		return *this;
	}

	inline void Allocate(const T& t){
		if(_ptr != NULL)
			delete _ptr;
		_ptr = new T(t);
	}

	inline T* Ptr(){return _ptr;}
	inline T* GetBuffer(){return _ptr;}
	inline const T& Ref(){return *(_ptr);}
};

//using namespace std;
//由vector构建2D数组
template< typename data_t >
class Vector2d_ptr
{	
	typedef data_t v_t;
	typedef std::vector< data_t > vec_t;
	typedef std::vector< vec_t* > vec_ptr;

	vec_ptr _vecPtrs;

public:
	Vector2d_ptr(){};
	Vector2d_ptr(vec_ptr& src):_vecPtrs(src)
	{}

	vec_ptr& Get2DPtr()
	{
		return _vecPtrs;
	}

	void AppendRow(vec_t* vptr)
	{
		_vecPtrs.push_back(vptr);
	}

	vec_t* operator()(int r)
	{
		return _vecPtrs.at(r);
	}

	data_t operator()(int r, int c)
	{
		return _vecPtrs.at(r)->at(c);
	}
};


//封装2D数组操作的类, 行优先存储
template<typename data_t>
class Array2d
{
	data_t* _data;
	size_t _col, _row, _size;

public:
	typedef std::vector<data_t> vec_t;
	typedef std::vector<vec_t*> vec_ptr;
	
	Array2d(): _row(0), _col(0), _size(0), _data(NULL)
	{}

	Array2d(vec_ptr& vptr)
	{
		_col = vptr[0]->size();
		_row = vptr.size();
		_size = _col * row;
		_data = new data_t[sizeof(data_t) * _size];

		data_t* start = 0;
		data_t* offset = 0;
		size_t data_len = 0;
		vec_t* index = 0;
		for(int r = 0; r < iRow; r++){
			vIndex = vptr[r];
			start = (data_t*)&vIndex->at(0);
			data_len = iCol * sizeof(data_t);
			offset = r * data_len;
			memcpy(_data + offset, start, data_len);			
		}
	}

	Array2d(size_t row, size_t col): _row(row), _col(col)
		, _size(row * col), _data(new data_t[sizeof(data_t) * (row * col)])
	{}

	Array2d& operator= (const Array2d& rhs)
	{
		if(this == &rhs)
			return *this;
		_row = rhs._row, _col = rhs._col, _size = rhs._size;
		if(_data != NULL)
			delete _data;
		_data = new data_t[sizeof(data_t) * _size];
		memcpy(_data, rhs._data, sizeof(data_t)*_size);
		return *this;
	}

	~Array2d()
	{
		if(_data != NULL)
			delete[] _data;
	}

	void Allocate(size_t r, size_t c){
		_row = r, _col = c, _size = r * c;
		if(_data != NULL)
			delete[] _data;
		_data = new data_t[sizeof(data_t) * _size];
	}

	int Row(){ return _row; }
	int Col(){ return _col; }

	bool Empty() { return (_size <= 0); }

	//先列后行?或先行后列?
	data_t& operator()(int ir, int ic)
	{
		return _data[ir * _col + ic];
	}
};


//---------------------------------------------------------------------------------------
//多维数组类, 按"列-行-页..."连续存储
template< class T >
class XArray
{
public:
	XArray():_totalSize(0), _dimLen(0), _dimVec(0), _pData(0){};

	//---------------------------------------------------------------------------------------
	//以一个存储了各维的维数的vector作为输入的构造函数
	XArray(std::vector<size_t>& vDim, char initVal = 0x0)
		: _dimLen(vDim.size()), _totalSize(1)
	{
		_dimVec = new size_t[_dimLen];
		_dimBuff = new size_t[_dimLen];
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			_dimVec[_ix] = vDim[_ix];
			_totalSize *= vDim[_ix];
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memset(_pData, initVal, _totalSize * sizeof(T));
	};

	//---------------------------------------------------------------------------------------
	//直接输入各维度, 语法: XArray(n, d1, d2, d3, ... dn)
	XArray(size_t dimLen, ...)
		: _dimLen(dimLen), _totalSize(1)
		, _dimVec(new size_t[dimLen])
		, _dimBuff(new size_t[dimLen])
	{		
		va_start(arg_ptr, dimLen);
		size_t dim = 0;
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			dim = va_arg(arg_ptr, size_t);
			if(dim == 0){
				_dimLen = _ix;
				return;
			}
			_dimVec[_ix] = dim;
			_totalSize *= dim;
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memset(_pData, 0x0, _totalSize * sizeof(T));
	};

	//---------------------------------------------------------------------------------------
	//直接输入各维度, 并用一块指定内存初始化, 语法: XArray(pdata, n, d1, d2, d3, ... dn)
	XArray(T* pdata, size_t dimLen, ...)
		: _dimLen(dimLen), _totalSize(1)
		, _dimVec(new size_t[dimLen])
		, _dimBuff(new size_t[dimLen])
	{		
		va_start(arg_ptr, dimLen);
		size_t dim = 0;
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			dim = va_arg(arg_ptr, size_t);
			if(dim == 0){
				_dimLen = _ix;
				return;
			}
			_dimVec[_ix] = dim;
			_totalSize *= dim;
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memcpy(_pData, pdata, _totalSize * sizeof(T));
	};

	//---------------------------------------------------------------------------------------
	//以存储了维度的数组及其长度作为输入
	XArray(const size_t dimlen, const size_t* dims)
		: _dimLen(dimLen), _totalSize(1)
		, _dimVec(new size_t[dimLen]), _dimBuff(new size_t[dimLen])
	{
		memcpy(_dimVec, dims, dimLen * sizeof(size_t));
		size_t dim = 0;
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			dim = dims[ix];
			if(dim == 0){
				_dimLen = _ix;
				return;
			}
			_totalSize *= dim;
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memset(_pData, 0x0, _totalSize * sizeof(T));
	};
	//---------------------------------------------------------------------------------------
	//以存储了维度的数组及其长度作为输入,并用指定内存块初始化
	XArray(T* pdata, const size_t dimlen, const size_t* dims)
		: _dimLen(dimLen), _totalSize(1)
		, _dimVec(new size_t[dimLen]), _dimBuff(new size_t[dimLen])
	{
		memcpy(_dimVec, dims, dimLen * sizeof(size_t));
		size_t dim = 0;
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			dim = dims[ix];
			if(dim == 0){
				_dimLen = _ix;
				return;
			}
			_totalSize *= dim;
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memcpy(_pData, pdata, _totalSize * sizeof(T));
	};

	//---------------------------------------------------------------------------------------
	//拷贝构造函数
	XArray(const XArray& xa)
		: _dimLen(xa._dimLen), _totalSize(xa._totalSize), _index(0), _ix(0)
		, _dimVec(new size_t[xa._dimLen]), _dimBuff(new size_t[xa._dimLen])
		, _pData(new T[xa._totalSize])
	{
		memcpy(_dimVec, xa._dimVec, _dimLen * sizeof(size_t));
		memcpy(_dimBuff, xa._dimBuff, _dimLen * sizeof(size_t));
		memcpy(_pData, xa._pData, _totalSize * sizeof(T));
	}

	//---------------------------------------------------------------------------------------
	//赋值运算符
	XArray& operator= (const XArray& xa)
	{
		if((XArray*)&xa == this)
			return *this;
		release();
		_dimLen = xa._dimLen, _totalSize = xa._totalSize, _index = _ix = 0;
		_dimVec = new size_t[_dimLen], _dimBuff = new size_t[_dimLen];
		_pData = new T[_totalSize];
		memcpy(_dimVec, xa._dimVec, _dimLen * sizeof(size_t));
		memcpy(_dimBuff, xa._dimBuff, _dimLen * sizeof(size_t));
		memcpy(_pData, xa._pData, _totalSize * sizeof(T));
		return *this;
	}

	~XArray()
	{
		release();
	};

	
	////////////////////////////////////////////////////////////////////////////////
	//assign赋值函数, 先清理原有内容再重塑
	//------------------------------------------------------------------------------
	bool Assign(T* pdata, size_t dimLen, ...)
	{
		va_start(arg_ptr, dimLen);
		if(dimLen < 1)
			return false;

		release();
		_dimLen = dimLen, _dimVec = new size_t[dimLen], _dimBuff = new size_t[dimLen];
		_totalSize = 1;
		size_t dim = 0;
		for(_ix = 0; _ix < _dimLen; ++_ix)
		{
			dim = va_arg(arg_ptr, size_t);
			if(dim == 0){
				_dimLen = _ix;
				release();
				return false;
			}
			_dimVec[_ix] = dim;
			_totalSize *= dim;
			_dimBuff[_ix] = (_ix == 0)?(1):(_dimBuff[_ix - 1] * _dimVec[_ix - 1]);
		}
		_pData = new T[_totalSize];
		memcpy(_pData, pdata, _totalSize * sizeof(T));
	};

	//访问函数, 依次传入各维度坐标
	inline T& operator() (const int d1, ...)
	{
		va_start(arg_ptr, d1);
		size_t dim = 0;
		_index = d1;
		for(_ix = 1; _ix < _dimLen; _ix++)
		{
			dim = va_arg(arg_ptr, int);
			_index += dim * _dimBuff[_ix];
		}
		return _pData[_index];
	};

	inline T& at(const std::vector<size_t>& coord)
	{
		_index = coord[0];
		for(_ix = 1; _ix < _dimLen; _ix++)
		{			
			_index += coord[_ix] * _dimBuff[_ix];
		}
		return _pData[_index];
	}

	//---------------------------------------------------------------------------------------
	//按行设置数组内容, 以CArray为容器
	template< typename CArray>
	inline void SetRow(CArray& src, size_t d2, ...)
	{
		va_start(arg_ptr, d2);
		size_t dim = 0;
		_index = d2*_dimBuff[1];
		for(_ix = 2; _ix < _dimLen; _ix++)
		{
			dim = va_arg(arg_ptr, size_t);
			_index += dim * _dimBuff[_ix];
		}
		for(_ix = 0; _ix < _dimVec[0]; _index++, _ix++)
			_pData[_index] = src[_ix];
	}

	//取得某行元素, 赋值语义
	inline std::vector<T> GetRow(size_t d2, ...)
	{
		va_start(arg_ptr, d2);
		size_t dim = 0;
		_index = d2 * _dimBuff[1];
		for(_ix = 2; _ix < _dimLen; _ix++)
		{
			dim = va_arg(arg_ptr, size_t);
			_index += dim * _dimBuff[_ix];
		}
		std::vector<T> res(_pData + _index, _pData + _index + _dimVec[0]);
		return res;
	}

	//取得某一行元素首地址, 返回其行的长度
	inline size_t GetRow(T** pptr, size_t d2, ...)
	{
		va_start(arg_ptr, d2);
		size_t dim = 0;
		_index = d2 * _dimBuff[1];
		for(_ix = 2; _ix < _dimLen; _ix++)
		{
			dim = va_arg(arg_ptr, size_t);
			_index += dim * _dimBuff[_ix];
		}		
		*pptr = _pData + _index;
		return _dimVec[0];
	}

	//取得某一行元素首地址及其行长度
	inline size_t GetRow(T** pptr, std::vector<int>& coord)
	{		
		size_t dim = 0;
		_index = coord[1] * _dimBuff[1];
		for(_ix = 2; _ix < _dimLen; _ix++)
		{
			dim = coord[_ix];
			_index += dim * _dimBuff[_ix];
		}		
		*pptr = _pData + _index;
		return _dimVec[0];
	}

	const size_t& GetDimLen(){return _dimLen;};
	const size_t* GetDimVec(){return _dimVec;};
	const size_t* GetDimBuff(){return _dimBuff;};

	const size_t& GetIndex(std::vector<size_t>& coord)
	{
		_index = coord[0];
		for(_ix = 1; _ix < _dimLen; _ix++)
		{			
			_index += coord[_ix] * _dimBuff[_ix];
		}
		return _index;
	}

private:
	va_list arg_ptr;
	int	_ix;
	size_t	_index;
	size_t	_dimLen;	
	size_t* _dimVec;
	size_t* _dimBuff;
	size_t	_totalSize;

	T* _pData;

	void release()
	{
		if(_dimVec){
			delete[] _dimVec;
			_dimVec = NULL;
		}
		if(_dimBuff){
			delete[] _dimBuff;
			_dimBuff = NULL;
		}
		if(_pData){
			delete[] _pData;
			_pData = NULL;
		}
	}
};

}
}
#endif