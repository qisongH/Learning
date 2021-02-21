# 目录

* [排序算法](#排序算法)
* * [桶排序](#桶排序)
* [二分查找](#二分查找)
* [TopK问题](#TopK问题)
* * [手写小(大)根堆](#手写小(大)根堆)
* [数组映射](#数组映射)
* [滑动窗口](#滑动窗口)
* * [模板[虫取法]](#模板[虫取法])





## 排序算法

### 桶排序

* *算法描述*

1、设置一个一定量的数组当作空桶；

2、遍历输入数据，并且把数据一个一个放到对应位置的桶里去；

3、对每个不是空的桶进行排序；

4、从不是空的桶里把排好序的数据拼接起来。

**两个关键点：桶的长度和桶的数量**

![](https://i.loli.net/2020/11/27/RqP4rGpBh3fuyaj.png)

* *算法思想*

> 桶排序的思想近乎彻底的**分治思想**。
>
> 桶排序假设待排序的一组数均匀独立的分布在一个范围中，并将这一范围划分成几个子范围（桶）。
>
> 然后基于某种映射函数f ，将待排序列的关键字 k 映射到第i个桶中 (即桶数组B 的下标i) ，那么该关键字k 就作为 B[i]中的元素 (每个桶B[i]都是一组大小为N/M 的序列 )。
>
> 接着将各个桶中的数据有序的合并起来 : 对每个桶B[i] 中的所有元素进行比较排序 (可以使用快排)。然后依次枚举输出 B[0]….B[M] 中的全部内容即是一个有序序列。
>
> 补充： 映射函数一般是 f = array[i] / k; k^2 = n; n是所有元素个数
>
> 为了使桶排序更加高效，我们需要做到这两点：
>
> 1. 在额外空间充足的情况下，尽量增大桶的数量
> 2. 使用的映射函数能够将输入的 N 个数据均匀的分配到 K 个桶中
>
> 同时，对于桶中元素的排序，选择何种比较排序算法对于性能的影响至关重要。
>
> 参考：https://blog.csdn.net/developer1024/article/details/79770240

显然桶的宽度和数量会极大的影响排序的效果，可以采用上述公式进行设计，也可以根据实际情况进行调整；

**比如一些极端的情况**

> 1. max-min = 100，len = 10000，那么按公式计算 bucketNum = 100 / 10000 + 1 = 1，也就是说只有一个桶，明显不符合常理，算法也就自动失效了；
>
> 2. 数组 [1, 100001]，max - min = 100000，len = 2，则 bucketNum = 100000 / 2 + 1，显然此时算法的空间复杂度爆炸。
>
> 参考：https://blog.csdn.net/qq_27124771/article/details/87651495



```C++
int bucketSort(vector<int> &nums)
{
	// 不需要排序
	if (nums.size() < 2)
		return 0;

	int len = nums.size();

	// 计算最大值和最小值
	int maxVal = *max_element(nums.begin(), nums.end());
	int minVal = *min_element(nums.begin(), nums.end());

	// 计算桶的宽度和数量
	int d = max(1, (maxVal - minVal) / (len - 1));
	int bucketSize = (maxVal - minVal) / d + 1;
	vector<list<int> > bucketNums(bucketSize);

	// 将每个元素放入桶
	for (int i = 0; i < len; ++i)
	{
		int idx = (nums[i] - minVal) / d;
		bucketNums[idx].push_back(nums[i]);
	}

	// 每个桶内的元素进行排序
	for (int i = 0; i < bucketSize; ++i)
	{
		if (bucketNums[i].size() == 0)
			continue;
		bucketNums[i].sort();
	}

	// 将桶中的值放回到原数组
	int index = 0;
	for (int i = 0; i < bucketSize; ++i)
	{
		for (int &bucketVal : bucketNums[i])	// 遍历桶内的每一个数据
		{
			nums[index++] = bucketVal;
		}
	}
	return 1;
}
```





* **例子1：**

  [leetcode 164](https://leetcode-cn.com/problems/maximum-gap/)

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 2，则返回 0。

**示例 1：**

```
输入: [3,6,9,1]
输出: 3
解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。
```

![](https://i.loli.net/2020/11/27/BiH9VpMn6uUAEkh.png)

```c++
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        if (nums.size() < 2)
            return 0;
        
        int minVal = *min_element(nums.begin(), nums.end());
        int maxVal = *max_element(nums.begin(), nums.end());
        int len = nums.size();

        int d = max(1, (maxVal - minVal) / (len - 1));  // 桶的宽度 (向下取整)
        int bucketSize = (maxVal - minVal) / d + 1;     // 桶的大小 (+1 保证最后一个数也能放)

        // 存桶 (桶内最小值，桶内最大值)对，空桶即为(-1, -1)
        vector<pair<int, int> > bucket(bucketSize, {-1, -1});
        for (int i = 0; i < len; ++ i)
        {
            int idx = (nums[i] - minVal) / d;   // 当前数在桶内的位置
            if (bucket[idx].first == -1)
            {
                bucket[idx].first = bucket[idx].second = nums[i];
            }
            else 
            {
                bucket[idx].first = min(bucket[idx].first, nums[i]);
                bucket[idx].second = max(bucket[idx].second, nums[i]);
            }
        } 

        int ansVal = 0, prev = -1;  // prev 记录前一个没空的桶
        for (int i = 0; i < bucketSize; ++ i)
        {
            if (bucket[i].first == -1)  // 说明是空桶
                continue;
            if (prev != -1)
                ansVal = max(ansVal, bucket[i].first - bucket[prev].second);
            prev = i;
        }
        return ansVal;
    }
};
```

可以证明，相邻的两个数必然大于 **桶的长度 d**，所以在将元素放入桶的同时，要维护桶内的最小值和最大值；

要找到相邻数字的间距，必然也是在两个不同（非空）的桶之间，不断的去比较相邻的两个桶，最大间距即为前一个桶的最大值和后一个桶的最小值之差；



* **例子2：**

[leetcode 1370](https://leetcode-cn.com/problems/increasing-decreasing-string/)

**示例 1**

```
输入：s = "aaaabbbbcccc"
输出："abccbaabccba"
解释：第一轮的步骤 1，2，3 后，结果字符串为 result = "abc"
第一轮的步骤 4，5，6 后，结果字符串为 result = "abccba"
第一轮结束，现在 s = "aabbcc" ，我们再次回到步骤 1
第二轮的步骤 1，2，3 后，结果字符串为 result = "abccbaabc"
第二轮的步骤 4，5，6 后，结果字符串为 result = "abccbaabccba"
```

```C++
class Solution {
public:
    string sortString(string s) {
        // 只关注字符本身，不用考虑字符在字符串中的位置
        // 用一个数组（桶）记录字符个数
        vector<int> num(26, 0);
        for (char &c : s)
        {
            num[c - 'a'] ++;
        }

        string str_ans = "";
        while (str_ans.size() != s.size())
        {
            // 最小的字符->最大的字符
            for (int i = 0; i < 26; ++ i)
            {
                if (num[i] > 0)
                {
                    str_ans.push_back(i + 'a');
                    num[i] --;
                }
            }
            // 最大的字符->最小的字符
            for (int i = 25; i >= 0; -- i)
            {
                if (num[i] > 0)
                {
                    str_ans.push_back(i + 'a');
                    num[i] --;
                }
            }
        }
        return str_ans;
    }
};
```

利用桶的概念，将每个字符映射到一个**数组**里，这个数组能够记录字符的**顺序和个数**；

再对这个数组*（桶）*从前往后和从后往前遍历即可，可以分别得到上升序列和下降序列；

## 二分查找

* **例子1：**

[leetcode 34](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

> 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
>
> 如果数组中不存在目标值 target，返回 [-1, -1]。

**示例 1**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

*针对有序数组，可以采用二分查找实现时间复杂度为 O(log n) 的算法*

本题分两步，一是找到目标值在数组中的起始位置，二是找到目标值在数组中的结束位置

```C++
    int FindFirstPos(vector<int> &nums, int target)
    {
        int len = nums.size();
        int low = 0, high = len - 1;
        // 先找到起始位置（高位逐次逼近）
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (nums[mid] < target)
            {
                low = mid + 1;
            }
            else if (nums[mid] == target)
            {
                high = mid; // 高位逐次逼近
            }
            else    // nums[mid] > target
            {
                high = mid - 1;
            }
        }
        // 不存在 target
        if (nums[low] != target)
            return -1;
        return low;
    }
```

```C++
    int FindLastPos(vector<int> &nums, int target)
    {
        int len = nums.size();
        int low = 0, high = len - 1;
        // 再确定结束位置（低位逐次逼近）
        while (low < high)
        {
            int mid = low + (high - low + 1) / 2;   // 注意这里的区别
            if (nums[mid] < target)
            {
                low = mid + 1;
            }
            else if (nums[mid] == target)
            {
                low = mid; // 低位逐次逼近
            }
            else    // nums[mid] > target
            {
                high = mid - 1;
            }
        }
        return low;
    }
```

## TopK问题

* **例子1：**

[leetcode 703](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

> 设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。
>
> 请实现 KthLargest 类：
>
> KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
> int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大的元素。

**示例 1**

```
输入：
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：
[null, 4, 5, 5, 8, 8]

解释：
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8

```

**思路与算法**

如果底层架构使用数组的话，每次调用 *add()* 函数，都需要向数组中添加一个元素，然后使用 *sort()* 进行排序，并返回排序后数组的第 *K* 个数字，其时间复杂度为 *O(K log(K))* 。

数组的核心问题是 **自身不带排序功能**，因此使用自带排序功能的数据结构——**堆**

![](https://i.loli.net/2021/02/11/syJdDiCMxkWg12w.png)

>本题的操作步骤如下：
>
>使用大小为 K 的**小根堆**，在初始化的时候，保证堆中的元素个数不超过 K 。
>在每次 ```add()``` 的时候，将新元素``` push()``` 到堆中，如果此时堆中的元素超过了 K，那么需要把堆中的最小元素（堆顶）```pop()``` 出来。
>此时堆中的最小元素（堆顶）就是整个数据流中的第 K 大元素。
>
>**问答：**
>1、为什么使用小根堆？
>因为我们需要在堆中保留数据流中的前 K 大元素，使用小根堆能保证每次调用堆的 ```pop()``` 函数时，从堆中删除的是堆中的最小的元素（堆顶）。
>2、为什么能保证堆顶元素是第 K 大元素？
>因为小根堆中保留的一直是堆中的前 K 大的元素，*堆的大小是 K*，所以堆顶元素是第 K 大元素。
>3、每次 ```add()``` 的时间复杂度是多少？
>每次 `add()` 时，调用了堆的 `push()` 和 `pop()` 方法，两个操作的时间复杂度都是 log(K)。

```C++
class KthLargest {
public:
    priority_queue<int, vector<int>, greater<int>> q;
    int k;
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (auto& x: nums) {
            add(x);
        }
    }
    
    int add(int val) {
        q.push(val);
        if (q.size() > k) {
            q.pop();
        }
        return q.top();
    }
};

```

### 手写小(大)根堆

[题解](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/solution/python-dong-hua-shou-xie-shi-xian-dui-by-ypz2/)

```C
/*
	堆 数据结构
	heap -> 存储数据的数组 heap[0]用于过渡
	heapSize -> 堆的元素个数
	cmp -> 小/大根堆的函数指针
*/
struct Heap {
    int* heap;
    int heapSize;
    bool (*cmp)(int, int);
};

/*
	初始化函数
*/
void init(struct Heap* obj, int n, bool (*cmp)(int, int)) {
    obj->heap = malloc(sizeof(int) * (n + 1));
    obj->heapSize = 0;
    obj->cmp = cmp;
}

/*
	比较函数
*/
bool cmp(int a, int b) {
    return a > b;
}

/*
	交换两个函数
*/
void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b, *b = tmp;
}

/*
	添加元素 同时需要调整元素的顺序（向上调整）
	1、首先把元素添加到数组的末尾，元素个数加1
	2、调整元素顺序，新元素(p)与其父节点(p/2)比较大小【p:元素索引】
	3、新元素小于父节点则交换位置（小根堆）
	4、重复步骤2、3直至结束
*/
void push(struct Heap* obj, int x) {
    int p = ++(obj->heapSize), q = p >> 1;
    obj->heap[p] = x;
    while (q) {
        if (!obj->cmp(obj->heap[q], obj->heap[p])) {
            break;
        }
        swap(&(obj->heap[q]), &(obj->heap[p]));
        p = q, q = p >> 1;
    }
}

/*
	弹出堆顶元素 同时需要调整元素的顺序（向下调整）
	1、首先把堆顶元素和尾元素交换，把元素个数减1（不是真正的弹出）
	2、调整元素顺序，新元素(p)与其孩子节点(p*2 和 p*2+1)比较大小【p:元素索引】
	3、新元素若小于孩子节点，则与其【较小】的孩子节点交换位置
	4、重复步骤2、3直至结束
*/
void pop(struct Heap* obj) {
    swap(&(obj->heap[1]), &(obj->heap[(obj->heapSize)--]));
    int p = 1, q = p << 1;
    while (q <= obj->heapSize) {
        if (q + 1 <= obj->heapSize) {
            if (obj->cmp(obj->heap[q], obj->heap[q + 1])) {
                q++;
            }
        }
        if (!obj->cmp(obj->heap[p], obj->heap[q])) {
            break;
        }
        swap(&(obj->heap[q]), &(obj->heap[p]));
        p = q, q = p << 1;
    }
}

/*
	返回堆顶元素
*/
int top(struct Heap* obj) {
    return obj->heap[1];
}

/*
	TOPK 结构体
*/
typedef struct {
    struct Heap* heap;
    int maxSize;
} KthLargest;

/*
	创建 TOPK 结构体
*/
KthLargest* kthLargestCreate(int k, int* nums, int numsSize) {
    KthLargest* ret = malloc(sizeof(KthLargest));	// 分配空间
    ret->heap = malloc(sizeof(struct Heap));
    
    init(ret->heap, k + 1, cmp);		// k+1 避免初始化时数组是空的情况
    ret->maxSize = k;
    for (int i = 0; i < numsSize; i++) {
        kthLargestAdd(ret, nums[i]);
    }
    return ret;
}

int kthLargestAdd(KthLargest* obj, int val) {
    push(obj->heap, val);
    if (obj->heap->heapSize > obj->maxSize) {
        pop(obj->heap);
    }
    return top(obj->heap);
}

void kthLargestFree(KthLargest* obj) {
    free(obj->heap->heap);		// 一级一级释放空间
    free(obj->heap);
    free(obj);
}

```



## 数组映射

[leetcode 566](https://leetcode-cn.com/problems/reshape-the-matrix/)

>给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。
>
>重构后的矩阵需要将原始矩阵的所有元素以相同的**行遍历顺序**填充。
>
>如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

```
输入: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
输出: 
[[1,2,3,4]]
解释:
行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。
```

**思路和算法**

![](https://i.loli.net/2021/02/17/9u5mevzsgL1QXMx.png)

```C++
class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) {
        int m = nums.size();
        int n = nums[0].size();
        if (m * n != r * c) {
            return nums;
        }

        vector<vector<int>> ans(r, vector<int>(c));
        for (int x = 0; x < m * n; ++x) {
            ans[x / c][x % c] = nums[x / n][x % n];
        }
        return ans;
    }
};

```



## 滑动窗口

**例子1：**

[leetcode955](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)

>在仅包含 0 和 1 的数组 A 中，一次 K 位翻转包括选择一个长度为 K 的（连续）子数组，同时将子数组中的每个 0 更改为 1，而每个 1 更改为 0。
>
>返回所需的 K 位翻转的最小次数，以便数组没有值为 0 的元素。如果不可能，返回 -1。
>

```
输入：A = [0,1,0], K = 1
输出：2
解释：先翻转 A[0]，然后翻转 A[2]。

输入：A = [1,1,0], K = 2
输出：-1
解释：无论我们怎样翻转大小为 2 的子数组，我们都不能使数组变为 [1,1,1]。

输入：A = [0,0,0,1,0,1,1,0], K = 3
输出：3
解释：
翻转 A[0],A[1],A[2]: A变成 [1,1,1,1,0,1,1,0]
翻转 A[4],A[5],A[6]: A变成 [1,1,1,1,1,0,0,0]
翻转 A[5],A[6],A[7]: A变成 [1,1,1,1,1,1,1,1]
```

**方法1：暴力模拟**

直接遍历数组A，当 *A[i]==0* 时，就把区间【i，i+K】全部进行翻转，直到 i+K == A.size()。**但是会超时，时间复杂度 O(NK)**

```C++
class Solution {
public:
    int minKBitFlips(vector<int>& A, int K) {
        int len = A.size(), ansMin = 0;
        int i = 0;
        
        while (i <= len - K)
        {
            if (A[i] == 1)
            {
                ++ i;
                continue;
            }

            for (int j = 0; j < K; ++ j)
            {
                A[i + j] = A[i + j] == 0 ? 1 : 0;
            }
            ++ i;
            ++ ansMin;
        }

        if (find(A.begin(), A.end(), 0) != A.end())
        {
            return -1;
        }
        else
            return ansMin;
    }
};
```

**方法2：差分数组**

根据方法1的思路，**考虑不去翻转数字，而是统计每个数字需要翻转的次数**；

一次翻转是对【i,i+K-1】区间的数都进行了改变，而对于区间操作，有一种常见的方法——**差分**（前缀和的逆运用，对区间两端进行操作以代替区间内操作）；

**差分数组diff[i]**：表示两个相邻元素A[i-1]和A[i]的**翻转次数的差**，对于区间【l，r】，将其元素全部加1（这个区间内所有元素翻转一次），只会影响 diff[l] 和 diff[r+1] 处的值，即 diff[l] + 1，diff[r+1] - 1；

再用一个变量 rcvCnt 记录当前位置（当前区间）翻转的次数。

```C++
class Solution {
public:
    int minKBitFlips(vector<int> &A, int K) {
        int n = A.size();
        vector<int> diff(n + 1);
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt += diff[i];
            if ((A[i] + revCnt) % 2 == 0) {
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                ++revCnt;
                --diff[i + K];
            }
        }
        return ans;
    }
};
```

模2意义下的加减法与异或等价，因此可以用异或改写代码。

```C++
class Solution {
public:
    int minKBitFlips(vector<int> &A, int K) {
        int n = A.size();
        vector<int> diff(n + 1);
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt ^= diff[i];
            if (A[i] == revCnt) { // A[i] ^ revCnt == 0
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                revCnt ^= 1;
                diff[i + K] ^= 1;
            }
        }
        return ans;
    }
};
```

*模拟过程*

```
输入：A = [0,0,0,1,0,1,1,0], K = 3

1、
i = 0, ans = 1, revCnt = 1,
diff = [0,0,0,1,0,0,0,0]
A_ = [1,1,1,1,0,1,1,0]

2、
i = 4, ans = 2, revCnt = 0(在i=3时，由1->0),
diff = [0,0,0,1,0,0,0,1]
A_ = [1,1,1,1,1,0,0,0]

3、
i = 5, ans = 3, revCnt = 1(在i=4时，由0->1),
diff = [0,0,0,1,0,0,0,1,(1)]
A_ = [1,1,1,1,1,1,1,1]
```

**方法3：滑动窗口**

上述方法中，数组 diff[] 用于记录区间的变化，revCnt 用于记录变化的次数，所以如果能知道位置 *i-K* 上发生了翻转操作，便可以直接修改 revCnt，从而去掉 diff[] 数组。

**所以可以利用原数组A[i]范围之外的数来表达【是否翻转过】的含义**

```C++
class Solution {
public:
    int minKBitFlips(vector<int> &A, int K) {
        int n = A.size();
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= K && A[i - K] > 1) {
                revCnt ^= 1;
                A[i - K] -= 2; // 复原数组元素，若允许修改数组 A，则可以省略
            }
            if (A[i] == revCnt) {
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                revCnt ^= 1;
                A[i] += 2;
            }
        }
        return ans;
    }
};

```



**示例2：**

[leetcode 1004](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

>给定一个由若干 `0` 和 `1` 组成的数组 `A`，我们最多可以将 `K` 个值从 0 变成 1 。
>
>返回仅包含 1 的最长（连续）子数组的长度。

```
输入：A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
输出：6
解释： 
[1,1,1,0,0,1,1,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 6。

输入：A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
输出：10
解释：
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 10。

```

**思路**

**重点：**题意转换。把「最多可以把 K 个 0 变成 1，求仅包含 1 的最长子数组的长度」转换为 **「找出一个最长的子数组，该子数组内最多允许有 K 个 0 」**。

经过上面的题意转换，我们可知本题是**求最大连续子区间**，可以使用滑动窗口方法。**滑动窗口的限制条件是：窗口内最多有 K 个 0**。

*代码思路：*

1、使用 left 和 right 两个指针，分别指向滑动窗口的**左右边界**。
2、right **主动**右移：right 指针每次移动一步。当 A[right] 为 0，说明滑动窗口内增加了一个 0；
3、left **被动**右移：判断此时窗口内 0 的个数，如果超过了 K，则 left 指针**被迫右移**，直至窗口内的 0 的个数小于等于 K 为止。
**滑动窗口长度的最大值就是所求**。

```c++
class Solution {
public:
    int longestOnes(vector<int>& A, int K) {
        int res = 0, zeros = 0, left = 0;
        for (int right = 0; right < A.size(); ++right) {
            if (A[right] == 0) ++zeros;
            while (zeros > K) {
                if (A[left++] == 0) --zeros;
            }
            res = max(res, right - left + 1);
        }
        return res;
    }
};
```

### 模板[虫取法]

《挑战程序设计竞赛》这本书中把**滑动窗口**叫做「虫取法」，我觉得非常生动形象。因为滑动窗口的两个指针移动的过程和虫子爬动的过程非常像：**前脚不动，把后脚移动过来；后脚不动，把前脚向前移动**。

```python
def findSubArray(nums):
    N = len(nums) # 数组/字符串长度
    left, right = 0, 0 # 双指针，表示当前遍历的区间[left, right]，闭区间
    sums = 0 # 用于统计 子数组/子区间 是否有效，根据题目可能会改成求和/计数
    res = 0 # 保存最大的满足题目要求的 子数组/子串 长度
    while right < N: # 当右边的指针没有搜索到 数组/字符串 的结尾
        sums += nums[right] # 增加当前右边指针的数字/字符的求和/计数
        while 区间[left, right]不符合题意：# 此时需要一直移动左指针，直至找到一个符合题意的区间
            sums -= nums[left] # 移动左指针前需要从counter中减少left位置字符的求和/计数
            left += 1 # 真正的移动左指针，注意不能跟上面一行代码写反
        # 到 while 结束时，我们找到了一个符合题意要求的 子数组/子串
        res = max(res, right - left + 1) # 需要更新结果
        right += 1 # 移动右指针，去探索新的区间
    return res
```

滑动窗口中用到了左右两个指针，它们移动的思路是：**以右指针作为驱动，拖着左指针向前走。右指针每次只移动一步，而左指针在内部 while 循环中每次可能移动多步。右指针是主动前移，探索未知的新区域；左指针是被迫移动，负责寻找满足题意的区间**。

模板的整体思想是：

1、定义两个指针 *left* 和 *right* 分别指向区间的开头和结尾，注意是闭区间；定义 *sums* 用来统计该区间内的各个字符出现次数；
2、第一重 while 循环是为了判断 *right* 指针的位置是否超出了数组边界；当 *right* 每次到了新位置，需要增加 *right* 指针的求和/计数；
3、第二重 while 循环是让 *left* 指针向右移动到 *[left, right]* **区间符合题意的位置**；当 *left* 每次移动到了新位置，需要**减少** *left* 指针的求和/计数；
4、在第二重 while 循环之后，成功找到了一个符合题意的 *[left, right]* 区间，题目要求最大的区间长度，因此更新 res 为 max(res, 当前区间的长度) 。
5、right 指针每次向右移动一步，开始**探索新的区间**。

模板中的 sums 需要根据题目意思具体去修改，本题是求和题目因此把sums 定义成整数用于求和；如果是计数题目，就需要改成字典用于计数。当左右指针发生变化的时候，都需要更新 sums 。

另外一个需要根据题目去修改的是内层 while 循环的判断条件，即： 区间 [left, right] 不符合题意 。对于本题而言，就是该区间内的 0 的个数 超过了 2 。



**示例3：**

[leetcode 1438](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

>给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。
>
>如果不存在满足条件的子数组，则返回 0 。

```
输入：nums = [8,2,4,7], limit = 4
输出：2 
解释：所有子数组如下：
[8] 最大绝对差 |8-8| = 0 <= 4.
[8,2] 最大绝对差 |8-2| = 6 > 4. 
[8,2,4] 最大绝对差 |8-2| = 6 > 4.
[8,2,4,7] 最大绝对差 |8-2| = 6 > 4.
[2] 最大绝对差 |2-2| = 0 <= 4.
[2,4] 最大绝对差 |2-4| = 2 <= 4.
[2,4,7] 最大绝对差 |2-7| = 5 > 4.
[4] 最大绝对差 |4-4| = 0 <= 4.
[4,7] 最大绝对差 |4-7| = 3 <= 4.
[7] 最大绝对差 |7-7| = 0 <= 4. 
因此，满足题意的最长子数组的长度为 2 。

输入：nums = [10,1,2,4,7,2], limit = 5
输出：4 
解释：满足题意的最长子数组是 [2,4,7,2]，其最大绝对差 |2-7| = 5 <= 5 。

输入：nums = [4,2,2,2,4,4,2,2], limit = 0
输出：3
```

**思路**

本题是求最大连续子区间，可以使用**滑动窗口**方法。滑动窗口的限制条件是：**窗口内最大值和最小值的差不超过 limit**。

如果遍历求滑动窗口内的最大值和最小值，时间复杂度是 *O(k)*，肯定会超时。**降低时间复杂度的一个绝招就是增加空间复杂度：利用更好的数据结构**。是的，我们的目的是快速让一组数据有序，那就寻找一个内部是有序的数据结构呗！下面我分语言讲解一下常见的内部有序的数据结构。

**在 C++ 中 set/multiset/map 内部元素是有序的，它们都基于红黑树实现。其中 set 会对元素去重，而 multiset 可以有重复元素，map 是 key 有序的哈希表。**

```C++
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        multiset<int> st;
        int left = 0, right = 0;
        int res = 0;
        while (right < nums.size()) {
            st.insert(nums[right]);
            while (*st.rbegin() - *st.begin() > limit) {
                st.erase(st.find(nums[left]));
                left ++;
            }
            res = max(res, right - left + 1);
            right ++;
        }
        return res;
    }
};

```

**可以使用双端队列，同时维护区间内的最大值和最小值**

```C++
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        int len = nums.size();
        int ansLen = 0, left = 0;
        deque<int> qmax, qmin;      // 降序队列、升序队列

        for (int right = 0; right < len; ++ right)
        {
            while (!qmax.empty() && nums[right] > qmax.back())  // 维护队列降序
                qmax.pop_back();
            while (!qmin.empty() && nums[right] < qmin.back())  // 维护队列升序
                qmin.pop_back();
            qmax.push_back(nums[right]);
            qmin.push_back(nums[right]);

            while (qmax.front() - qmin.front() > limit)
            {
                if (nums[left] == qmax.front())
                    qmax.pop_front();
                if (nums[left] == qmin.front())
                    qmin.pop_front();
                ++ left;
            }
            ansLen = max(ansLen, right - left + 1);
        }
        return ansLen;
    }
};
```

