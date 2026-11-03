# 第一周周报 
## 学习内容
- markdown 语法
- c语言流程控制语句，函数，做了几个小练习
- 机器学习：初步学习梯度下降  
- --
## markdown语法

## c语言
- switch语句，if语句
- case穿透,for,while循环（break,continue,goto）
- 函数
## 梯度下降
- 以代价函数J=(w,b)为例：    
 求min J(w,b):利用数学公式每次稍微改变参数值，减小J,直到J最终稳定或接近最小值
 - 数学公式       
 $$w=w-\alpha \frac{\partial}{\partial w} J(w,b)$$        
 $$b=b-\alpha \frac{\partial}{\partial b} J(w,b)$$\
    其中 $\alpha$ 为称为learn rate,用来控制步长，为0到1的小正数，不能太小或太大
    太大：w,b无法收敛
    太小：步长小，效率慢
- $\omega$,b需要同时更新，才能达到最好效果\
  CORRECT:\
 $$tmp_w=w-\alpha \frac{\partial}{\partial w} J(w,b)$$\
 $$tmp_b=b-\alpha \frac{\partial}{\partial b} J(w,b)$$\
 $$w=tmp_w$$\
 $$b=tmp_b$$\
  FALSE:\
  $$tmp_w=w-\alpha \frac{\partial}{\partial w} J(w,b)$$\
 $$w=tmp_w$$  

>  这一步w已经更新了

$$tmp_b=b-\alpha \frac{\partial}{\partial b} J(w,b)$$       
$$b=tmp_b$$ 

