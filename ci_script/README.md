ci_script 脚本目录结构
====
假设项目名称为**CI_PROJECT_NAME** (对于本例为hs_image_io), 则目录下部署如下脚本:

- {CI_PROJECT_NAME}_cmake_generate.sh: *项目的cmake生成脚本*
- {CI_PROJECT_NAME}_install.sh: *项目安装脚本*
- {CI_PROJECT_NAME}_build_test.sh: *项目单元测试生成脚本*
- {CI_PROJECT_NAME}_run_test.sh: *项目测试运行脚本.*

同时, 在子目录hslib_ci_script/ 下, 包含如下脚本文件:

- build_test.sh: *测试用例生成脚本*
- cmake_generate.sh: *具体cmake生成脚本*
- initialize.sh: *初始化脚本*
- install.sh: *生成安装脚本*
- resolve_dependency.sh: *依赖项解析及生成脚本*
- run_test.sh: *测试运行脚本*

各脚本说明
====
{CI_PROJECT_NAME}_cmake_generate.sh
----
用于项目的cmake生成操作.

**主要逻辑**

1.调用 hslib_ci_script/**initialize.sh** , 完成相关变量初始化及导出.

2.调用 hslib_ci_script/**resolve_dependency.sh** 进行依赖项的CI生成.

例如, 若项目含依赖项gtest, 则在该脚本中需要调用 **resolve_dependency.sh** : 

```
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" "yong" "gtest"
```

3.调用 hslib_ci_script/**cmake_generate.sh** 完成cmake生成操作.

**脚本接收参数**

+ 具名参数

--build_shared: 是否编译为共享库, 若不指定该参数则编译为静态库(static).

--build_types *type_values*: cmake编译选项字符串, 其中可指定多个选项, 以分号(;)分隔, 例如`"Debug;Release;RelWithDebInfo"`. 若未指定该参数, 则根据 *build_type* 参数确定编译选项及对应的生成目录.

+ 非具名参数

*build_type* : 紧随 *--build_shared* 或 *--build_types* 后的第一个参数, 指定CI的编译生成选项(Debug, Release 或 RelWithDebInfo).

*generator* : 紧随 *build_type* 后的第一个参数, 指定平台相关的编译生成类型, 目前可用的值为: `"Unix Makefiles"`, `"Visual Studio 12 Win64"`, `"Visual Studio 12"`.

例如使用64位的VS2013编译为共享库, CI编译类型为Debug, 则可按以下方式调用:

```
./hs_image_io_cmake_generate.sh \
  --build_shared --build_types "Debug;" "Debug" "Visual Studio 12 Win64"
```
  
{CI_PROJECT_NAME}_install.sh
----
直接进行项目的安装生成操作

**主要逻辑**

1.调用hslib_ci_script/**initialize.sh** , 完成相关变量初始化及导出.

2.调用 hslib_ci_script/**install.sh** , 根据指定编译选项完成INSTALL项目生成.

**脚本接收参数**

同 {CI_PROJECT_NAME}_cmake_generate.sh

{CI_PROJECT_NAME}_build_test.sh
----
直接生成项目的单元测试项目.

**主要逻辑**

1.调用hslib_ci_script/**initialize.sh** , 完成相关变量初始化及导出.

2.调用 hslib_ci_script/**build_test.sh** , 根据指定编译选项完成单元测试项目生成. 例如对于子项目stream_io, 生成单元测试用例为stream_io_utest, 则需调用 **build_test.sh**:

```
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/build_test.sh \
  "stream_io_utest"
```

**脚本接收参数**

同 {CI_PROJECT_NAME}_cmake_generate.sh

{CI_PROJECT_NAME}_run_test.sh
----
运行指定的单元测试

**主要逻辑**

1.调用hslib_ci_script/**initialize.sh** , 完成相关变量初始化及导出.

2.调用 hslib_ci_script/**run_test.sh** , 进入指定单元测试路径运行可执行文件.

**脚本接收参数**

同 {CI_PROJECT_NAME}_cmake_generate.sh

