import  numpy as np
a = [[1,1,1],
     [2,2,2],
     [3,3,3]
     ]
i,j=np.shape(a)
z=1
for c in range(i):
    for b in range(j):
        z = z*a[c][b]
c=np.power(z,(1/9))
print(int(c))