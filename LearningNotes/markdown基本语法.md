# markdown常用语法
## **一、标题**
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
## **二、字体**
*倾斜*  （两个空格换行）  
**加粗**  
***倾斜加粗***  
~~删除线~~  
<u>下划线</u>  

## **三、列表**
+ 无序列表
    + 无序列表
        + 无序列表

- 无序列表
    - 无序列表
        - 无序列表

1. 有序列表
2. 有序列表
3. 有序列表
4. 

## **四、表格**
| 星期一 | 星期二 | 星期三 | 星期四 | 星期五 | 星期六 | 星期日 | 
| ----:| ---- | :----: | ---- | ---- | ---- | ---- |
| 摆烂 | 摆烂  | 摆烂 | 摆烂 | 摆烂 | 摆烂 | 摆烂 |

## **五、代码**
```python
import torch
```

```c
#include <stdio.h>
```
行内代码：`print(a)`

## **六、LaTex公式**
LaTeX 是一种高质量的排版格式，可以生成复杂的表格与数学公式，是当前电子与数学出版行业的事实标准。可前往以下网站查询具体用法。  
https://www.zybuluo.com/codeep/note/163962  

行内公式：$f(x) = a + b$  

行间公式：
$$
    f(x) = a + b
$$

常见公式命令 

上下标：   $ a^2 $、$ a_1 $  

分数：$\frac{1}{2}$

求和：$\sum_{i=1}^{m}$  

编号：
$$
    a+b=c\tag{0}
$$

$$
    J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2\tag{1}
$$
![](https://cdn.jsdelivr.net/gh/huaqianyue/imgs@master/markdown-imgs/202211182043226.png)
![](https://cdn.jsdelivr.net/gh/huaqianyue/imgs@master/markdown-imgs/202211191844176.png)
![](https://cdn.jsdelivr.net/gh/huaqianyue/imgs@master/markdown-imgs/202211191934755.png)