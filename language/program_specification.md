# 目录

* [初始化顺序](#初始化顺序)
* [条件判断](#条件判断)
* [switch语句](#switch语句)
* [自增自减运算](#自增自减运算)
* [栈溢出](#栈溢出)
* [引用](#引用)
* [const](#const)
* [数组退化](#数组退化)
* [函数与宏](#函数与宏)
* [类的成员](#类的成员)
* [虚函数](#虚函数)
* [析构函数](#析构函数)
* [内存操作函数](#内存操作函数)





## 初始化顺序

![](https://i.loli.net/2021/05/24/dmVPY97WoglihJD.png)

**不同文件中的全局变量，其初始化顺序是不确定的，所以调用时有可能不是预期的结果。**

![](https://i.loli.net/2021/05/24/znWcuXx8d3lrtoS.png)

**推荐写法：**

对需要在另一个文件中调用的全局变量用**函数封装**。



## 条件判断

![](https://i.loli.net/2021/05/24/6liVtRvC1ZmP95Q.png)



## switch 语句

![](https://i.loli.net/2021/05/24/UneikyFgJstMYoA.png)



## 自增自减运算

![](https://i.loli.net/2021/05/24/On3Sj2twQlobHKP.png)





## 栈溢出

![](https://i.loli.net/2021/05/24/B8Kr5NZlX42tyAV.png)

局部变量大小：**1K（经验值）**



## 引用

![](https://i.loli.net/2021/05/24/V3vT8uzOlnyoJh4.png)

传入的形参为指针时，应当在函数体内首先**判断指针是否为空**（**建议使用引用&**）；



## const

![](https://i.loli.net/2021/05/24/ItEq3mFMoYXkVTU.png)



## 数组退化

![](https://i.loli.net/2021/05/24/2KRUx9PoJBN81Ev.png)

形参为数组类型时，**传入的都是指针**；



## 函数和宏

![](https://i.loli.net/2021/05/24/Iz4t9Fr8qbnK2S7.png)



## 类的成员

![](https://i.loli.net/2021/05/24/SXORzvimw1ZPeq3.png)

**所有成员都应当正确初始化，否则易访问到未定义的值**。



## 虚函数

![](https://i.loli.net/2021/05/24/UGPrc8NJMSztBA2.png)

**虚函数的重写：必须是完全一样（函数名、形参、返回值等）**。



## 析构函数

![](https://i.loli.net/2021/05/24/kc6mJtvRhGxgZB9.png)



## 内存操作函数

![](https://i.loli.net/2021/05/24/MVCUThznFS5pXdm.png)

**POD对象：如C语言中的struct（只有变量无函数）**

