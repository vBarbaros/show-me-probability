import numpy as np
"""
#generate a dictionary with all possible outcomes of rolling two dices
d = {(i,j):i+j for i in range(1,7) for j in range(1,7)}

from collections import defaultdict

#generating a dictionary of lists, each list contains all the combinations that can make up for the sum j
dinv = defaultdict(list)
for i,j in d.iteritems():
	dinv[j].append(i)
	
#generating the probability mass function
X = {i:len(j)/36. for i,j in dinv.iteritems()}

for i,j in X.iteritems():
	print i, ": ", j

#what is the probability that half the product of three dice will exceed their sum?
d = {(i,j,k):((i*j*k)/2 > i+j+k) for i in range(1, 7)
									for j in range(1, 7)
										for k in range(1, 7)}

dinv = defaultdict(list)
for i,j in d.iteritems():
	dinv[j].append(i)

X = {i:len(j)/6.0**3 for i,j in dinv.iteritems() }

print X

from pandas import DataFrame
#index = [(i,j) for i in range(1,7) for j in range(1,7)]
#columns = ['sm', 'd1', 'd2', 'pd1', 'pd2', 'p']

d = DataFrame(index=[(i,j) for i in range(1,7) for j in range(1,7)], columns=['sm', 'd1', 'd2', 'pd1', 'pd2', 'p'])

d.d1=[i[0] for i in d.index]
d.d2=[i[1] for i in d.index]

d.sm=map(sum, d.index)
print d.head(6)

d.loc[d.d1<=3, 'pd1'] = 1/9.
d.loc[d.d1 >3, 'pd1'] = 2/9.

d.pd2 = 1/6.
print d.head(10)

d.p = d.pd1 * d.pd2
print d.head(6)

print d.groupby('sm')['p'].sum()
"""


x = np.arange(30)
print x
x.shape = (3, 10)
print x

print x[0:2+1, 0:2+1]

print x[0:3, 1:4]

print x[0:3, 2:5]


































