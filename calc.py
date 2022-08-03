import math
import numpy
f = open("score.txt","rt")
scores=[]
while True:
    line = f.readline()
    if line =='':
        break
    scores.append(float(line)) 

f.close()
mean=numpy.mean(scores)
var=numpy.var(scores)
std=numpy.std(scores)
print("평균 : ",mean)
print("분산 : ",var)
print("표준편차 : ",std)

val=float(input("값을 입력해주세요"))

if val >= (mean+1.5*std):
    print("300")
elif (mean+1.5*std) > val >= mean +std:
    print("100")
elif mean +std > val >= mean + 0.5 *std:
    print("60")
elif mean + 0.5 *std > val >= mean:
    print("40")
elif mean > val >= mean - 0.5* std:
    print("20")
elif mean- 1.5* std > val :
    print("10")
else:
    print("0") 