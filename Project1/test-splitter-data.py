from sklearn.model_selection import train_test_split
import os
dummyLabels=[0 for x in range(len(os.listdir('./data/video1/W6')))]
X_train, X_test, y_train, y_test = train_test_split(os.listdir('./data/video1/W6'), dummyLabels, test_size=0.1)

for i in range(len(X_test)):    
    os.rename('./data/video1/W6/'+X_test[i],'./data/video1/W6_test/'+X_test[i])