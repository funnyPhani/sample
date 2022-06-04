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



```python
try:
    import os
    os.makedirs("repImg",exist_ok=True)
    import urllib      
    import requests
    from bs4 import BeautifulSoup
    import warnings
    
    def downloadImage(url,name):
        urllib.request.urlretrieve(url,name)

    warnings.filterwarnings("ignore")
    url = "https://indianrecipes.com/new_and_popular"
#     url = f"https://indianrecipes.com/api?tm={time.time()}"
    req = requests.get(url).content
    soup = BeautifulSoup(req,"html.parser")
    data = soup.findAll("div",attrs={"class":"links group"})
    for i in data:
        for j in i.findAll("a",{"class":"group"}):
            for n in j.findAll("div",{"class":"text"}):
                print(n.text.strip())
                try:
                    os.makedirs(os.path.join("repImg",n.text.strip()),exist_ok=True)
                except Exception as e:
                    print("error occured :",e)
            print("https:"+j.get("href"))
            purl = "https:"+j.get("href")
            for k in j.findAll("div",{"class":"image"}):
                for l in k.findAll("picture"):
                    for m in l.findAll("source"):
                            print("https:"+m.get("srcset"))  
                            img = "https:"+m.get("srcset")
                            downloadImage(img,f"repImg/{n.text.strip()}/{n.text.strip()}.jpg")
            
            req1 = requests.get(purl).content
            soup1 = BeautifulSoup(req1,"html.parser")
            data1 = soup1.findAll("div",{"class":'instructions'})
            d = ""
            for o in data1:
                d+=o.text.strip()
                d = "".join(d).strip().replace("\n"," ")
                print(d)
            with open(f"repImg/{n.text.strip()}/{n.text.strip()}.txt","w") as f:
                f.writelines(f"Name :{n.text.strip()}"+"\n")
                f.writelines(f"URL :{purl}"+"\n")
                f.writelines(f"ImageURL :{img}"+"\n")
                f.writelines(f"Description :{d}"+"\n")

            print("-"*95)
except Exception as e:
    print("error occured :",e)
```
