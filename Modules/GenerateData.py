import random
import pandas as pd

X = []
for i in range(10000):
    x = random.random()
    y = random.random()
    if(y > x + 0.2):
        l = 0
    else:
        l = 1
    X.append([x, y, l])

X = pd.DataFrame(X)
X.to_csv("Data.csv", index = False)