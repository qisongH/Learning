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



![](https://i.loli.net/2021/06/15/A2B7iW8wpLgrDzJ.png)

void* 转换类似C语言中的强制转换，它不会进行语法检查，是不安全的行为；

C++ 中的 dynamic_cast<type> 转换具有类型检查的功能；

*BaseA、BaseB* 的指针指向的地址不一样，动态转换过程中会对其进行相应的偏移。



## 类的成员

![](https://i.loli.net/2021/05/24/SXORzvimw1ZPeq3.png)

**所有成员都应当正确初始化，否则易访问到未定义的值**。



## 虚函数

![](https://i.loli.net/2021/05/24/UGPrc8NJMSztBA2.png)

**虚函数的重写：必须是完全一样（函数名、形参、返回值等）**。



![](https://i.loli.net/2021/06/15/DR7AXFwQGkhVl8c.png)

调用的函数是运行时决定，**而缺省参数是编译时决定的**。



## 析构函数

![](https://i.loli.net/2021/05/24/kc6mJtvRhGxgZB9.png)



![](https://i.loli.net/2021/06/15/P6oLcwMuCgbkJDN.png)



## 内存操作函数

![](https://i.loli.net/2021/05/24/MVCUThznFS5pXdm.png)

**POD对象：如C语言中的struct（只有数据成员无任何构造函数、析构函数）**



## Lambda表达式

![](https://i.loli.net/2021/06/15/7C5oW1Nd9KyMFre.png)

*lambda* 表达式 [=] 一般是在上下文按值捕获，理论上 *data++* 不应该改变成员变量的值；

但是在上述程序中 [=] **会捕获 *this* 指针**，*data++* 会改变成员变量的值；

若仍想要以值捕获的形式传递参数，应当如右图所示**复制 *this* 的副本**。 



![](https://i.loli.net/2021/06/15/jkEW4nyiVzeKGJX.png)



## new操作符

![](https://i.loli.net/2021/06/15/x8Q4rfyW7hpvBbq.png)

*new* 操作符不会返回空指针（即使失败）；

应当**使用 C++ 提供的异常处理机制**；



## 智能指针

![](https://i.loli.net/2021/06/15/nZXQv3eP1um7fTF.png)

问题1：重复写两次 MyClass 类型会**增加维护成本和时间**；

问题2：若第二个 new MyClass 出现异常，会导致第一个 new 发生**内存泄漏**；

**解决方法**：应当使用 C++ 集成好的创建智能指针函数；



## 迭代器

![](https://i.loli.net/2021/06/15/eJOjsM3vfBX7Vmb.png)

问题：增删元素会导致迭代器（vector、list、map等）失效！

解决方法：使用 STL 自带的函数；



## 头文件

![](https://i.loli.net/2021/06/16/tYu7Egy4sNMlSqh.png)

![](https://i.loli.net/2021/06/16/UceQ29sRdNG1LMD.png)

*main.cpp* 中重复包含 *a.h* 两次，可能导致编译出错，同一个变量可能会被定义两次；

应当采用宏进行保护；



## 代码风格约定

![](https://i.loli.net/2021/06/16/rFhHb752oitNGA3.png)

![](https://i.loli.net/2021/06/16/GnNBD1qOuoEpKde.png)

![](https://i.loli.net/2021/06/16/Yi4nuQdMeVWgrwC.png)

![](https://i.loli.net/2021/06/16/9Fgf5IjHJtVTnqv.png)

