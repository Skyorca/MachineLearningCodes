################################

#SVM with simplified-SMO
#SkyOrca, UCAS, 2017.10.3

#note: Kernel = Self, K(x, x) = x*x.T

#################################


from numpy import *

def loadData(filepath):
    dataMatrix = []; classMatrix = []
    fp = open(filepath)
    for line in fp.readlines():
        lineArry = line.strip().split('\t')
        dataMatrix.append([float(lineArry[0]),float(lineArry[1])])  #(x,y)
        classMatrix.append(float(lineArry[2]))
    return dataMatrix, classMatrix

def selectJrand(i, m):
    j = i
    while(i == j):
        j = int(random.uniform(0,m))
    return j
# each iteration choose two alpha
def adjustAlpha(alpha, H, L):
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha
#control the value of alpha (0<=alpha<=C,loosen?)
'''
def simplified_SMO(dataMatrixIn, classMatrixIn, C, toler, maxIter):
    dataMatrix = np.mat(dataMatrixIn); classMatrix = np.mat(classMatrixIn).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alpha = np.mat(np.zeros((m, 1)))
    iter = 0

    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            R_i = float(np.multiply(alpha, classMatrix).T*\
                       (dataMatrix*dataMatrix[i, :].T)) + b
            E_i = R_i - float(classMatrix[i])

            if (classMatrix[i]*E_i < -toler and alpha[i] < C) or (classMatrix[i]*E_i > toler and alpha[i]>0):
                j = selectAnotherAlpha(i, m)
                R_j = float(np.multiply(alpha, classMatrix).T*\
                         dataMatrix*dataMatrix[j, :].T) + b
                E_j = R_j - float(classMatrix[j])

                alpha_i_Old = alpha[i].copy()
                alpha_j_Old = alpha[j].copy()

                if classMatrix[i] != classMatrix[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C+alpha[j]-alpha[i])
                else:
                    L = max(0, alpha[j]+alpha[i]-C)
                    H = min(C, alpha[j]+alpha[i])
                if L == H: print 'L==H'; continue    # L==H indicates no changes, alpha is OK

                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0: print 'eta>=0'; continue
                alpha[j] -= classMatrix[j]*(E_j - E_i)/eta
                alpha[j] = adjustAlpha(alpha[j], H, L)
                if abs(alpha[j] - alpha_j_Old) < 0.00001: print 'not moving enough!'; continue
                alpha[i] += classMatrix[i]*classMatrix[j]*(alpha_j_Old - alpha[j])

                b_1 = b - E_i - classMatrix[i]*(alpha[i] - alpha_i_Old)*dataMatrix[i, :]*dataMatrix[i, :].T\
                              - classMatrix[j]*(alpha[j] - alpha_j_Old)*dataMatrix[j, :]*dataMatrix[j, :].T
                b_2 = b - E_j - classMatrix[i]*(alpha[i] - alpha_i_Old)*dataMatrix[i, :]*dataMatrix[j, :].T\
                              - classMatrix[j]*(alpha[j] - alpha_j_Old)*dataMatrix[j, :]*dataMatrix[j, :].T
                if alpha[i] > 0 and alpha[i] < C:   b =b_1
                elif alpha[j] > 0 and alpha[j] < C: b = b_2
                else:                               b = (b_1 + b_2)/2.0
                alphaPairsChanged += 1
                print 'iter: %d, alpha_i: %d, pairs changed: %d, ' % (iter, i, alphaPairsChanged)
            if alphaPairsChanged == 0: iter += 1
            print 'iternum: %d' % iter
    return alpha, b
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = adjustAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

class optStruct:
    def __init__(self, dataMatrixIn, classMatrixIn, C, toler):
        self.X = dataMatrixIn
        self.label = classMatrixIn
        self. C = C
        self.toler = toler
        self.m = shape(dataMatrixIn)[0]
        self.b = 0
        self.alphas = mat(zeros((self.m, 1)))
        self.eCache = mat(zeros((self.m, 2)))

def calcEk(os, k):
    res = float(multiply(os.alphas, os.label).T * (os.X*os.X[k, :].T))+os.b
    Ek = res - os.label[k]
    return Ek
def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]

def selectJ(i, os, Ei):
    maxJ = -1;maxDeltaE = 0;Ej = 0
    os.eCache[i] = [1, Ei]
    validEcacheList = nonzero(os.eCache[:, 0].A)[0]
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(os, k)
            if(abs(Ek - Ei)>maxDeltaE):
                maxDeltaE = abs(Ek - Ei); maxJ = k; Ej = Ek
        return maxJ, Ej
    else:
        maxJ = selectJrand(i, os.m)
        Ej   = calcEk(os, maxJ)
    return maxJ, Ej

def innerProcess(i, os):
    Ei = calcEk(os, i)
    if ((os.label[i]*Ei < -os.toler) and (os.alphas[i] < os.C)) or ((os.label[i]*Ei > os.toler) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i,os, Ei)
        alphaIold = os.alphas[i].copy(); alphaJold = os.alphas[j].copy();
        if (os.label[i] != os.label[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * os.X[i,:]*os.X[j,:].T - os.X[i,:]*os.X[i,:].T - os.X[j,:]*os.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        os.alphas[j] -= os.label[j]*(Ei - Ej)/eta
        os.alphas[j] = adjustAlpha(os.alphas[j],H,L)
        updateEk(os, j)
        if (abs(os.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        os.alphas[i] += os.label[j]*os.label[i]*(alphaJold - os.alphas[j])#update i by the same amount as j
                                                                #the update is in the oppostie direction
        updateEk(os, i)
        b1 = os.b - Ei- os.label[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[i,:].T - os.label[j]*(os.alphas[j]-alphaJold)*os.X[i,:]*os.X[j,:].T
        b2 = os.b - Ej- os.label[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[j,:].T - os.label[j]*(os.alphas[j]-alphaJold)*os.X[j,:]*os.X[j,:].T
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):    os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):  os.b = b2
        else: os.b = (b1 + b2)/2.0
        return 1
    else:return 0

def SMO(dataMatrixIn, classMatrixIn, C, toler, maxIter):
    os = optStruct(mat(dataMatrixIn), mat(classMatrixIn).transpose(), C, toler)
    iter = 0; alphaPairsChanged = 0; entireSet = 1
    while (iter<maxIter) and ((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerProcess(i, os)
                print 'fullSet, iter %d, i %d, pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerProcess(i,os)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return os.b,os.alphas


def calcW(alphas, dataMatrixIn, classMatrixIn):
    X = mat(dataMatrixIn)
    labelMat = mat(classMatrixIn).transpose()
    m, n = shape(X)
    W = zeros((n, 1))
    for i in range(m):
        W += multiply(alphas[i]*labelMat[i], X[i, :].T)
    return W



def execu_2(filepath):
    dataArr, labelArr = loadData(filepath)
    b, alphas = SMO(dataArr, labelArr, 0.6, 0.001, 40)
    W = calcW(alphas, dataArr, labelArr)
    dataMat = mat(dataArr)
    m = shape(dataMat)[0]
    ResList = []
    for i in range(m):
        Res = dataMat[i]*mat(W) + b
        if Res[0] > 0 : ResList.append(1)
        else          : ResList.append(-1)
        if ResList[i] == labelArr[i]:
            print 'Right!'
        else:
            print 'Wrong!'

    '''print 'b:', b
    print 'alphas:', alphas
    print 'W:', W
    print 'Finished'
    '''

execu_2("/home/skyfish/DL/ML/srccode/Ch06/testSet.txt")

































































'''
def execu_1(filepath):
    dataMatrix, labelMatrix = loadData(filepath)
    alpha, b = smoSimple(dataMatrix, labelMatrix, 0.6, 0.001, 40)
    print 'Showing Results......'
    print 'b: ', b
    print 'alpha: ', alpha[alpha > 0]
    print 'number of support vectors: %d' % shape(alpha[alpha > 0])
    print 'Printing support vectors:'
    for i in range(100):
        if alpha[i] > 0.0: print dataMatrix[i], labelMatrix[i]
    print 'Finished'

execu("/home/skyfish/DL/ML/srccode/Ch06/testSet.txt")
'''
