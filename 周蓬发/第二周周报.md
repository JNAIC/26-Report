# 第二周周报

## 学习内容

- python类的创建及使用，文件和异常处理
- 深度学习预备知识：数据操作，线性代数，微分概率等； 

---

## python

- 类

  ``` python
  class Car:
      def __init__(self, make, model, year):
          self.make = make
          self.model = model
          self.year = year
          self.odometer_reading = 0
  
      def get_descriptive_name(self):
          long_name = f"{self.year} {self.make} {self.model}"
          return long_name.title()
  
  
  class ElectricCar(Car):
      def __init__(self, make, model, year):
          super().__init__(make, model, year)
  
  
  my_leaf = ElectricCar('nissan', 'leaf', 2024)
  print(my_leaf.get_descriptive_name()) ```
  ```
  

---

## 深度学习

~~跟着书学的，本来预备知识想跳过，后来发现高估自己了~~

- 主要为数学知识：降维，张量算法的性质，点积，矩阵，范数，自动微分及其代码实现

  



