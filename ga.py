import knn
import numpy

def generatePop(n):
	chromAsli=[]
	for x in range(n):
		temp=[]
		temp2=[]
		for x in range(10):
			temp.append(numpy.random.randint(1))
		temp2.append(temp)
		fitness=knn.fitnessValue(temp)
		temp2.append(fitness)
		chromAsli.append(temp2)
	return chromAsli

def eliteChild(chrom2,n):
	a=sorted(chrom2,key=lambda l:l[1], reverse=True)
	bestVal=a[0][1]
	bestFt2=a[0][0]
	next2=[]
	chrom3=[]
	for x in range(2):
		next2.append(a[x][0])
	for x in range(2,n):
		chrom3.append(a[x])
	return chrom3,next2,bestVal,bestFt2

def tournament(chrom2):
    best=[]
    for x in range(2):
        a=numpy.random.randint(0,7)
        best.append(chrom2[a])
    bestOne=best[0]
    if(best[0][1]<best[1][1]):
        bestOne=best[1]
    return bestOne[0]

def getMutation(chrom2,pm,next2,n):
    rng=n-len(next2)
    for i in range(rng):
        a=tournament(chrom2)
        rd=[]
        for x in range(10):
            ru=numpy.random.uniform(0.0, 1.0)
            if ru<=pm:
                rd.append(1) 
            else:
                rd.append(0)
        if a==rd:
            a=tournament(chrom2)
        result=toXor(a,rd)
        next2.append(result)
    return next2

def toXor(x1,x2):
	xorRes=[]
	for x in range(len(x1)):
		if x1[x]==x2[x]:
			xorRes.append(0)
		else:
			xorRes.append(1)
	return xorRes

def getCrossOver(chrom2,pr,next2):
	rng=int(pr*len(chrom2))
	for x in range(rng):
		a=tournament(chrom2)
		b=tournament(chrom2)
		if a==b:
			result=a
		else:
			result=toXor(a,b)
		next2.append(result)
	return next2

def getGeneration(chrom2,idx):
	chrom2,nextGen,bestFitness,bestFt2=eliteChild(chrom2,genNum)
	nextGen=getCrossOver(chrom2,0.8,nextGen)
	nextGen=getMutation(chrom2,0.3,nextGen,genNum)
	print('Generation {} {:.3f}'.format(idx+1,float(bestFitness)),end='%\n')
	return nextGen,bestFitness,bestFt2

genNum=15
knn_chrom=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
knn_acc=knn.fitnessValue(knn_chrom)
limit=120
stall=50
chrom=generatePop(genNum)
counter=0
newFit=0.0
oldFit=0.0
for x in range(limit):
    oldFit=newFit
    newChrom,newFit,bestFt=getGeneration(chrom,x)
    newFit=float(newFit)
    temp2=[]
    for y in range(genNum):
        temp=[]
        temp.append(newChrom[y])
        f=knn.fitnessValue(newChrom[y])
        temp.append(f)
        temp2.append(temp)
    chrom=temp2
    diff=newFit - oldFit
    if diff<=0.0000001:
        counter+=1
    else:
    	counter=0
    if counter==stall:
    	break
    
print("Accuracy using K-NN without GA: ",end='')
print('{:.3f}'.format(float(knn_acc)),end='')
print("%\nAccuracy using K-NN with GA:",end='')
print('{:.3f}'.format(float(newFit)),end='%')
print("\nUsed features:", bestFt)