本项目是计算大数相乘的`NTT`实现（`GPU`）

`input.zip`解压后将`intput.txt`放到根目录，`intput.txt`是一个一百多MB的文本文件，文本一共有两行，每行六千多万位数字，程序计算后会在根目录输出`output.txt`存储计算结果

`ntt_gpu.cu`是`cuda`文件，可以直接使用`nvcc ntt_gpu.cu`编译后运行可执行文件

`gmp_cacl.cpp`是使用`gmp`大数库计算两数相乘

提示：

1、运行程序前确保你的电脑已经配置好了`cuda c++`环境或者`gmp`库

2、`gpu`与`cpu`的时间均只统计计算时间

3、代码很乱，尤其是`cuda`代码，还有很多优化空间