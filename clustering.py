import numpy as np
import matplotlib.pyplot as plt

np.random.seed(53)
def load_data(fname, type = None):
    features = []
    
    with open(fname) as F:
        
        for line in F:
            p = line.strip().split(' ')
            if type == "animals":
                p.append(10)
            elif type == "countries":
                p.append(11)
            elif type == "fruits":
                p.append(12)
            else:
                p.append(13)
            features.append(np.array(p[1:], float))
    return np.array(features)

dataset1 = load_data("animals", type = "animals")
dataset2 = load_data("countries", type = "countries")
dataset3 = load_data("fruits", type = "fruits")
dataset4 = load_data("veggies", type = "veggies")
dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4))

X = dataset[:,:-1]
y = dataset[:,-1]

class KMeans(object):

    def __init__(self, dataset, K = 4, maximumIter = 150):
        self.X = dataset
        self.K = K
        self.maxIterations = maximumIter
        self.centroids = []
        for i in range(K):
            emptyClustersList = []
        self.clusters = [emptyClustersList]
       
    def fit(self):   
        indices = np.random.choice(self.X.shape[0], self.K, replace=False)
        self.centroids = np.array(list([self.X[indices_index] for indices_index in indices]))
        

        for i in range(self.maxIterations):            
            self.clusters = self.clustersMaker(self.centroids)[0]
            cprior = self.centroids
            self.centroids = self.formCenters(self.clusters)           
            if self.movementChecker(self.centroids, cprior):
                break
        
        return self.centroids, self.clustersMaker(self.centroids)[1]

    def formCenters(self, clusters):
        centroids = np.empty((self.K, self.X.shape[1]))
        for clusterIndex, cluster in enumerate(clusters):
            cMean = np.mean(self.X[cluster], axis = 0)
            centroids[clusterIndex] = cMean

        return centroids

    def clustersMaker(self,centroids):
        clusters = []
        for i in range(self.K):
            emptyClusters = []
            clusters.append(emptyClusters) 
        
        for objIndex, objects in enumerate(self.X):
            centroidIndices = self.shortestDist(objects, centroids)
            clusters[centroidIndices].append(objIndex)
        clusterID = np.zeros(self.X.shape[0])

        for clusterIndex, cluster in enumerate(clusters):
            for objIndex in cluster:
                clusterID[objIndex] = clusterIndex      
        
        return clusters, clusterID

    def movementChecker(self, centroids, cprior):
        distances = []
        for i in range(self.K):
           dist = self.eucDist(centroids[i],cprior[i])
           distances.append(dist)
           
        if sum(distances) == 0:
            return True
        else:
            return False

    def shortestDist(self, objects, centroids):
        
        distancesList = []
        for c in centroids:
            dist = self.eucDist(c,objects)            
            distancesList.append(dist)
        indexOfNearest = np.argmin(distancesList)
       
        return indexOfNearest#, mydict

    def eucDist(self, X, Y):
        return np.sqrt(np.sum((X - Y)**2))
    

    def getBcubedPrecision(self, alldataset):
        self.individualp = []
        for i in range(self.K):
            clustersize = np.count_nonzero(alldataset[:,-1] == i)

            from10ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 10))            
            for j in range(from10ini):
                indp = from10ini/clustersize
                self.individualp.append(indp)

            from11ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 11))            
            for j in range(from11ini):
                indp = from11ini/clustersize
                self.individualp.append(indp)

            from12ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 12))            
            for j in range(from12ini):
                indp = from12ini/clustersize
                self.individualp.append(indp)

            from13ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 13))            
            for j in range(from13ini):
                indp = from13ini/clustersize
                self.individualp.append(indp)      
        bcubedprecision =  sum(self.individualp)/len(alldataset)                        
              
        return bcubedprecision,self.individualp

    def getBcubedRecall(self, alldataset):
        self.individualr = []
        
        for i in range(self.K):
            totalnum10 = np.count_nonzero(alldataset[:,-2] == 10)
            from10ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 10))            
            from10noti = totalnum10 - from10ini
            for j in range(from10ini):
                indp = from10ini/(from10ini + from10noti)
                self.individualr.append(indp)
                           
            totalnum11 = np.count_nonzero(alldataset[:,-2] == 11)
            from11ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 11))
            from11noti = totalnum11 - from11ini
            for k in range(from11ini):
                indp = from11ini/(from11ini + from11noti)
                self.individualr.append(indp)
                 
            totalnum12 = np.count_nonzero(alldataset[:,-2] == 12)
            from12ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 12))
            from12noti = totalnum12 - from12ini
            for l in range(from12ini):
                indp = from12ini/(from12ini + from12noti)
                self.individualr.append(indp)
            

            totalnum13 = np.count_nonzero(alldataset[:,-2] == 13)
            from13ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 13))
            from13noti = totalnum13 - from13ini
            for m in range(from13ini):
                indp = from13ini/(from13ini + from13noti)
                self.individualr.append(indp)
        bcubedrecall = sum(self.individualr)/len(alldataset)
                          
        return bcubedrecall, self.individualr
  

class KMedians(object):
    
    def __init__(self, dataset, K = 4, maximumIter = 150):
        self.X = dataset
        self.K = K
        self.maxIterations = maximumIter
        self.centroids = []
        for i in range(K):
            emptyClustersList = []
        self.clusters = [emptyClustersList]       
       
    def fit(self):   
        indices = np.random.choice(self.X.shape[0], self.K, replace=False)
        self.centroids = np.array(list([self.X[indices_index] for indices_index in indices]))

        for i in range(self.maxIterations):            
            self.clusters = self.clustersMaker(self.centroids)[0]
            cprior = self.centroids
            self.centroids = self.formCenters(self.clusters)           
            if self.movementChecker(self.centroids, cprior):
                break
        
        return self.centroids, self.clustersMaker(self.centroids)[1]

    def formCenters(self, clusters):
        centroids = np.empty((self.K, self.X.shape[1]))
        for clusterIndex, cluster in enumerate(clusters):
            cMedian = np.median(self.X[cluster], axis=0)
            centroids[clusterIndex] = cMedian

        return centroids

    def clustersMaker(self,centroids):
        clusters = []
        for i in range(self.K):
            emptyClusters = []
            clusters.append(emptyClusters) 
        
        for objIndex, objects in enumerate(self.X):
            centroidIndices = self.shortestDist(objects, centroids)
            clusters[centroidIndices].append(objIndex)

        clusterID = np.zeros(self.X.shape[0])
        for clusterIndex, cluster in enumerate(clusters):
            for objIndex in cluster:
                clusterID[objIndex] = clusterIndex      
        
        return clusters, clusterID, 

    def movementChecker(self, centroids, cprior):
        distances = []
        for i in range(self.K):
           dist = self.manDist(centroids[i],cprior[i])
           distances.append(dist)
           
        if sum(distances) == 0:
            return True
        else:
            return False

    def shortestDist(self, objects, centroids):
        distancesList = []
        for c in centroids:
            dist = self.manDist(c,objects)
            distancesList.append(dist)
        indexOfNearest = np.argmin(distancesList)
        return indexOfNearest  

    def manDist(self, X, Y):    
        return np.abs(X - Y).sum()

    def mdgetBcubedPrecision(self, alldataset):
        self.individualp = []
        for i in range(self.K):
            clustersize = np.count_nonzero(alldataset[:,-1] == i)

            from10ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 10))            
            for j in range(from10ini):
                indp = from10ini/clustersize
                self.individualp.append(indp)

            from11ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 11))            
            for j in range(from11ini):
                indp = from11ini/clustersize
                self.individualp.append(indp)

            from12ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 12))            
            for j in range(from12ini):
                indp = from12ini/clustersize
                self.individualp.append(indp)

            from13ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 13))            
            for j in range(from13ini):
                indp = from13ini/clustersize
                self.individualp.append(indp)      
        bcubedprecision =  sum(self.individualp)/len(alldataset)                        
              
        return bcubedprecision,self.individualp

    def mdgetBcubedRecall(self, alldataset):
        self.individualr = []
        
        for i in range(self.K):
            totalnum10 = np.count_nonzero(alldataset[:,-2] == 10)
            from10ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 10))            
            from10noti = totalnum10 - from10ini
            for j in range(from10ini):
                indp = from10ini/(from10ini + from10noti)
                self.individualr.append(indp)
                           
            totalnum11 = np.count_nonzero(alldataset[:,-2] == 11)
            from11ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 11))
            from11noti = totalnum11 - from11ini
            for k in range(from11ini):
                indp = from11ini/(from11ini + from11noti)
                self.individualr.append(indp)
                 
            totalnum12 = np.count_nonzero(alldataset[:,-2] == 12)
            from12ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 12))
            from12noti = totalnum12 - from12ini
            for l in range(from12ini):
                indp = from12ini/(from12ini + from12noti)
                self.individualr.append(indp)
            

            totalnum13 = np.count_nonzero(alldataset[:,-2] == 13)
            from13ini = np.count_nonzero((alldataset[:,-1] == i) & (alldataset[:,-2] == 13))
            from13noti = totalnum13 - from13ini
            for m in range(from13ini):
                indp = from13ini/(from13ini + from13noti)
                self.individualr.append(indp)
        bcubedrecall = sum(self.individualr)/len(alldataset)
                          
        return bcubedrecall, self.individualr

def dataExtender(X,y,clusterIDs):    
    datawithlabel = np.column_stack((X, y))
    extendedData = np.column_stack((datawithlabel , clusterIDs))
    return extendedData

def dataNormalizer(dataset):
    dataset = np.asarray(dataset)
    rows, columns = dataset.shape
    holder = np.zeros((rows, columns))
    holder = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    return holder

def getBcubedFscore(alldataset, means = True):
    if means:
        precision = np.asarray(kmns.getBcubedPrecision(alldataset)[1])
        recall = np.asarray(kmns.getBcubedRecall(alldataset)[1])
        numerator = 2*precision*recall
        denominator = precision + recall         
        result = np.sum(numerator/denominator)
        fscore = result/len(alldataset)
        return fscore
    else:
        precision = np.asarray(kmds.mdgetBcubedPrecision(alldataset)[1])
        recall = np.asarray(kmds.mdgetBcubedRecall(alldataset)[1])
        numerator = 2*precision*recall
        denominator = precision + recall         
        result = np.sum(numerator/denominator)
        fscore = result/len(alldataset)
        return fscore

print("************************************K-Means unnormalized Data**********************************************")

grandPrecision = []
grandRecall = []
grandFscore = []
for i in range(1,10):
    kmns = KMeans(X, K = i)
    result = kmns.fit()
    labels = result[1]
    fulldata = dataExtender(kmns.X,y,labels)    
    precision = kmns.getBcubedPrecision(fulldata)[0]
    grandPrecision.append(round(precision,2))
    recall = kmns.getBcubedRecall(fulldata)[0]
    grandRecall.append(round(recall, 2))
    fscore = getBcubedFscore(fulldata)
    grandFscore.append(round(fscore, 2))

    print("Precision for",i,"clusters is:", precision)
    print("Recall for",i,"clusters is:", recall)
    print("F-score for",i,"clusters is:",fscore)

    print("\n")
print("K-Means precision List for K = 1 to 9, respectively:",grandPrecision)
print("K-Means Recall List for K = 1 to 9, respectively:",grandRecall)
print("K-Means F-score List for K = 1 to 9 respectively:",grandFscore)

print("\n")
print("*********************************K-Means Normalized Results***********************************************")

normgrandPrecision = []
normgrandRecall = []
normgrandFscore = []
normalized = dataNormalizer(X)
for i in range(1,10):
    normkmns = KMeans(normalized, K = i)
    normresult = normkmns.fit()
    normlabels = normresult[1]
    normfulldata = dataExtender(normkmns.X,y,normlabels)    
    normprecision = normkmns.getBcubedPrecision(normfulldata)[0]
    normgrandPrecision.append(round(normprecision,2))
    normrecall = normkmns.getBcubedRecall(normfulldata)[0]
    normgrandRecall.append(round(normrecall, 2))
    normfscore = getBcubedFscore(normfulldata)
    normgrandFscore.append(round(normfscore, 2))

    print("K-Means Normalised Precision for",i,"clusters is:", normprecision)
    print("K-Means Normalised Recall for",i,"clusters is:", normrecall)
    print("K-Means Normalised F-score for",i,"clusters is:",normfscore)

    print("\n")
print("K-Means Normalised precision List for K = 1 to 9, respectively:",normgrandPrecision)
print("K-Means Normalised Recall List for K = 1 to 9, respectively:",normgrandRecall)
print("K-Means Normalised F-score List for K = 1 to 9 respectively:",normgrandFscore)

print("\n")

print("******************************* K-Medians Results*************************************")
mdgrandPrecision = []
mdgrandRecall = []
mdgrandFscore = []
for i in range(1,10):
    kmds = KMedians(X, K = i)
    kmdsresult = kmds.fit()
    kmdslabels = kmdsresult[1]
    kmdsfulldata = dataExtender(kmds.X,y,kmdslabels)    
    kmdsprecision = kmds.mdgetBcubedPrecision(kmdsfulldata)[0]
    mdgrandPrecision.append(round(kmdsprecision,2))
    kmdsrecall = kmds.mdgetBcubedRecall(kmdsfulldata)[0]
    mdgrandRecall.append(round(kmdsrecall, 2))
    kmdsfscore = getBcubedFscore(kmdsfulldata, means = False)
    mdgrandFscore.append(round(kmdsfscore, 2))

    print("K-Medians precision for",i,"clusters is:", kmdsprecision)
    print("K-Medians Recall for",i,"clusters is:", kmdsrecall)
    print("K-Medians F-score for",i,"clusters is:",kmdsfscore)

    print("\n")
print("K-Medians Precision List for K = 1 to 9, respectively:",mdgrandPrecision)
print("K-Medians Recall List for K = 1 to 9, respectively:",mdgrandRecall)
print("K-Medians F-score List for K = 1 to 9 respectively:",mdgrandFscore)

print("\n")


print("******************************* K-Medians Normalised Results*************************************")
normmdgrandPrecision = []
normmdgrandRecall = []
normmdgrandFscore = []
kmdsnormalized = dataNormalizer(X)
for i in range(1,10):
    normkmds = KMedians(kmdsnormalized, K = i)
    normkmdsresult = normkmds.fit()
    normkmdslabels = normkmdsresult[1]
    normkmdsfulldata = dataExtender(normkmds.X,y,normkmdslabels)    
    normkmdsprecision = normkmds.mdgetBcubedPrecision(normkmdsfulldata)[0]
    normmdgrandPrecision.append(round(normkmdsprecision, 2))
    normkmdsrecall = normkmds.mdgetBcubedRecall(normkmdsfulldata)[0]
    normmdgrandRecall.append(round(normkmdsrecall, 2))
    normkmdsfscore = getBcubedFscore(normkmdsfulldata, means = False)
    normmdgrandFscore.append(round(normkmdsfscore, 2))

    print("K-Medians Normalised precision for",i,"clusters is:", normkmdsprecision)
    print("K-Medians Normalised Recall for",i,"clusters is:", normkmdsrecall)
    print("K-Medians Normalised F-score for",i,"clusters is:",normkmdsfscore)

    print("\n")
print("K-Medians Normalised Precision List for K = 1 to 9, respectively:",normmdgrandPrecision)
print("K-Medians Normalised Recall List for K = 1 to 9, respectively:",normmdgrandRecall)
print("K-Medians Normalised F-score List for K = 1 to 9 respectively:",normmdgrandFscore)

print("\n")

# print("******************************* Plots and graphs*************************************")

# # Question 3
# x1 = [i for i in range(1, 10)]
# y1 = grandPrecision

# x2 = [i for i in range(1, 10)]
# y2 = grandRecall

# x3 = [i for i in range(1, 10)]
# y3 = grandFscore

# plt.plot(x1, y1, label = "Precision")
# plt.plot(x2, y2, label = "Recall")
# plt.plot(x3, y3, label = "F-score")

# plt.xlabel('Clusters(K)')
# plt.ylabel('Computed Values')
# plt.title('K-Means')
# plt.legend()
# plt.show()

# # Question 4

# xx1 = [i for i in range(1, 10)]
# yy1 = normgrandPrecision

# xx2 = [i for i in range(1, 10)]
# yy2 = normgrandRecall

# xx3 = [i for i in range(1, 10)]
# yy3 = normgrandFscore

# plt.plot(xx1, yy1, label = "Precision")
# plt.plot(xx2, yy2, label = "Recall")
# plt.plot(xx3, yy3, label = "F-score")

# plt.xlabel('Clusters(K)')
# plt.ylabel('Computed Values')
# plt.title('K-Means Normalised')
# plt.legend()
# plt.show()

# # Question 5

# xxx1 = [i for i in range(1, 10)]
# yyy1 = mdgrandPrecision

# xxx2 = [i for i in range(1, 10)]
# yyy2 = mdgrandRecall

# xxx3 = [i for i in range(1, 10)]
# yyy3 = mdgrandFscore

# plt.plot(xxx1, yyy1, label = "Precision")
# plt.plot(xxx2, yyy2, label = "Recall")
# plt.plot(xxx3, yyy3, label = "F-score")

# plt.xlabel('Clusters(K)')
# plt.ylabel('Computed Values')
# plt.title('K-Medians')
# plt.legend()
# plt.show()

# # Question 6

# xxxx1 = [i for i in range(1, 10)]
# yyyy1 = normmdgrandPrecision

# xxxx2 = [i for i in range(1, 10)]
# yyyy2 = normmdgrandRecall

# xxxx3 = [i for i in range(1, 10)]
# yyyy3 = normmdgrandFscore

# plt.plot(xxxx1, yyyy1, label = "Precision")
# plt.plot(xxxx2, yyyy2, label = "Recall")
# plt.plot(xxxx3, yyyy3, label = "F-score")

# plt.xlabel('Clusters(K)')
# plt.ylabel('Computed Values')
# plt.title('K-Medians Normalised')
# plt.legend()
# plt.show()



# fig, axs = plt.subplots(2,2)

# # KMeans unnormalised data

# axs[0, 0].plot(range(1, kmns.K +1), grandPrecision, marker='x', label='Precision')
# axs[0, 0].plot(range(1, kmns.K +1), grandRecall, marker='^', label='Recall')
# axs[0, 0].plot(range(1, kmns.K +1), grandFscore, marker='o', label='F-score')
# axs[0, 0].set_ylim([0.3, max(grandPrecision) + 0.1])
# axs[0, 0].set_title('K-Means')

# # KMeans Normalised data

# axs[0, 1].plot(range(1, normkmns.K +1), normgrandPrecision, marker='x', label='Precision')
# axs[0, 1].plot(range(1, normkmns.K +1), normgrandRecall, marker='^', label='Recall')
# axs[0, 1].plot(range(1, normkmns.K +1), normgrandFscore, marker='o', label='F-score')
# axs[0, 1].set_ylim([0.3, max(normgrandPrecision) + 0.1])
# axs[0, 1].set_title('K-Means Normalised')


# # KMedians unnormalised data

# axs[1, 0].plot(range(1, kmds.K +1), mdgrandPrecision, marker='x', label='Precision')
# axs[1, 0].plot(range(1, kmds.K +1), mdgrandRecall, marker='^', label='Recall')
# axs[1, 0].plot(range(1, kmds.K +1), mdgrandFscore, marker='o', label='F-score')
# axs[1, 0].set_ylim([0.3, max(mdgrandPrecision) + 0.1])
# axs[1, 0].set_title('K-Medians')


# # K Medians Normalised Data

# axs[1, 1].plot(range(1, normkmds.K +1), normmdgrandPrecision, marker='x', label='Precision')
# axs[1, 1].plot(range(1, normkmds.K +1), normmdgrandRecall, marker='^', label='Recall')
# axs[1, 1].plot(range(1, normkmds.K +1), normmdgrandFscore, marker='o', label='F-score')
# axs[1, 1].set_ylim([0.3, max(normmdgrandPrecision) + 0.1])
# axs[1, 1].set_title('K-Medians Normalised')

# for ax in axs.flat:
#     ax.set(xlabel='Cluster Sizes , K', ylabel='Computed Scores')
# for ax in axs.flat:
#     ax.label_outer()
# for ax in axs.flat:
#     ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1.35), ncol=3)  
#     break  
# plt.show()