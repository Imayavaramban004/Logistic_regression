import pandas as pd
from sklearn.model_selection import train_test_split as TTS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score,classification_report
data=pd.read_csv("C:\\Users\\imaya\\Desktop\\New folder\\imaya\\logistic regression\\candy-data.csv")
model=LR(class_weight='balanced')
# 1. Candy Type Prediction
# Problem Statement: Can you predict whether a candy contains chocolate based on other features such as fruity, caramel, peanutyalmondy, nougat, crispedricewafer, etc.?
# Target Variable: chocolate (1 if the candy contains chocolate, 0 otherwise)
# Potential Approach: Binary classification using logistic regression to determine the presence of chocolate.
xdata_1=data.drop(['competitorname','chocolate'],axis=1)
ydata_1=data['chocolate']
x1_train,x1_test,y1_train,y1_test=TTS(xdata_1,ydata_1,test_size=0.3,random_state=42)
model1_fit=model.fit(x1_train,y1_train)
y1_predict=model1_fit.predict(x1_test)
print("Accuracy: ",accuracy_score(y1_test,y1_predict))
print(classification_report(y1_test,y1_predict))
# 2. Candy Popularity Prediction
# Problem Statement: Can you predict whether a candy is popular among consumers based on its ingredients and other attributes?
# Target Variable: Convert winpercent into a binary variable (e.g., 1 if winpercent > 70, 0 otherwise).
# Potential Approach: Binary classification to determine if a candy is popular.
xdata_2=data.drop(['competitorname','winpercent'],axis=1)
ydata_2=(data['winpercent']>70).astype('int')
x2_train,x2_test,y2_train,y2_test=TTS(xdata_2,ydata_2,test_size=0.3,random_state=42)
model2_fit=model.fit(x2_train,y2_train)
y2_predict=model.predict(x2_test)
print("Accuracy: ",accuracy_score(y2_test,y2_predict))
print(classification_report(y2_test,y2_predict))