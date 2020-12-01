# 目录

* [排序算法](#排序算法)
* * [桶排序](#桶排序)
* [二分查找](#二分查找)





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

