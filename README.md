# sample
```python
!pip install pyforest
import pyforest
df = sns.load_dataset("iris")
df.head()

display(df['species'].value_counts())

df1 = df[df['species'] == "setosa"].sample(10)
df2 = df[df['species'] == "versicolor"].sample(2)
df3 = df[df['species'] == "virginica"].sample(5)
df4 = pd.concat([df1,df2,df3])
df4.reset_index(drop=True,inplace=True)
df4['species'].value_counts()

from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(k_neighbors=1)
X_bal, y_bal = over_sampler.fit_resample(df4.iloc[:,:-1], df4.iloc[:,-1])

display(y_bal.value_counts())

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_bal,y_bal,test_size=0.2,random_state=25)
from sklearn.linear_model import LinearRegression
lr = LogisticRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(ytest,ypred)
accuracy_score(ytest,ypred)
print(classification_report(ytest,ypred))
sns.heatmap(confusion_matrix(ypred,ytest),annot = True,xticklabels = df4['species'].unique(),yticklabels = df4['species'].unique());
```
