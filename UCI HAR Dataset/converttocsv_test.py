import numpy as np
import pandas as pd
import csv

# read input to a list of lists
dataContent = [i.strip().split() for i in open("./test/X_test.txt").readlines()]
dataContent = np.asarray(dataContent)
activity_labels = [i.strip().split() for i in open("./activity_labels.txt").readlines()]
yContent = [i.strip().split() for i in open("./test/y_test.txt").readlines()]
activity = []
for i in yContent:
	activity.append(activity_labels[int(i[0])-1][1])
activity = np.asarray(activity)
activity = np.resize(activity,(2947,1))
subContent = [i.strip().split() for i in open("./test/subject_test.txt").readlines()]
subContent = np.asarray(subContent,dtype='int32')
features = [i.strip().split() for i in open("./features.txt").readlines()]
features.append(['562','subject'])
features.append(['563','Activity'])
features = np.asarray(features)
_ , features = np.hsplit(features,2)
features = features.T
r = np.concatenate((dataContent,subContent,activity),axis=1)
r = np.concatenate((features,r),axis=0)

# write it as a new CSV file
with open("./test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(r)
