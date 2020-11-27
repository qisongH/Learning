# 目录

* [排序算法](# 排序算法)
* * [桶排序](# 桶排序)





## 排序算法

### 桶排序

* *算法描述*

1、设置一个一定量的数组当作空桶；

2、遍历输入数据，并且把数据一个一个放到对应位置的桶里去；

3、对每个不是空的桶进行排序；

4、从不是空的桶里把排好序的数据拼接起来。

**两个关键点：桶的长度和桶的数量**

![](https://i.loli.net/2020/11/27/RqP4rGpBh3fuyaj.png)

* 例子1：

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

可以证明，相邻的两个数必然大于 **桶的长度 d**，所以一个桶内最多只会有两个数；

要找到相邻数字的间距，必然也是在两个不同（非空）的桶之间；



