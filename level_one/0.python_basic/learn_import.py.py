import time
import numpy
import random as r

from functions import *

a = r.randint(1,5)
b = r.randint(1,5)



tik = time.time()

data = int(input(f"请输入{a} - {b} 的结果"))

result = sub(a,b)

if data == result:
    print("恭喜，回答成功")
else:
    print("回答错误！")


tok = time.time()

print(tok - tik)
