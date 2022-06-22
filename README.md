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


```python

# Day-1 web-scrapping
try:
    !pip install bs4
    import requests
    from bs4 import BeautifulSoup
    url = "https://indianrecipes.com/"
    # get the data from the url
    req = requests.get(url).content
    # parser the content from the requested url
    soup = BeautifulSoup(req,"html.parser")
#     print(soup.prettify())
    data = soup.findAll("div",{"class":"links group"})
#     print(data.text)
    for i in data:
        for j in i.findAll("a",{"class":"group"}):
            print("Dish Name :",j.text.strip())
            print("URL :","https:"+j.get("href"))
            for k in j.findAll("div",{"class":"image"}):
                for l in k.findAll("picture"):
                    for m in l.findAll("source"):
                        print("Image URL :","https:"+m.get("srcset"))
                        break
                    print("-"*105)
except Exception as e:
    print("error occured :",e)
    
```
  
```python
# Day-2 web-scrapping
try:
#     !pip install bs4
    import requests
    import os
    from bs4 import BeautifulSoup
    import urllib
     # new folder syntax
#     os.makedirs("vantalu",exist_ok=True)
    
    def downloadImage(url,name):
        urllib.request.urlretrieve(url,name)
    url = "https://indianrecipes.com/new_and_popular"
#     url = "https://indianrecipes.com/"
    # get the data from the url
    req = requests.get(url).content
    # parser the content from the requested url
    soup = BeautifulSoup(req,"html.parser")
#     print(soup.prettify())
    data = soup.findAll("div",{"class":"links group"})
#     print(data.text)
    for i in data:
        for j in i.findAll("a",{"class":"group"}):
            print("Dish Name :",j.text.strip())
            os.makedirs(os.path.join("vantalu",j.text.strip()),exist_ok=True)
            print("URL :","https:"+j.get("href"))
            for k in j.findAll("div",{"class":"image"}):
                for l in k.findAll("picture"):
                    for m in l.findAll("source"):
                        print("Image URL :","https:"+m.get("srcset"))
                        downloadImage("https:"+m.get("srcset"),f"vantalu/{j.text.strip()}/{j.text.strip()}.jpg")
                        break
                    print("-"*105)
except Exception as e:
    print("error occured :",e)
```


```python
try:
    import requests
    from bs4 import BeautifulSoup
    req = requests.get("http://books.toscrape.com/").content
    soup = BeautifulSoup(req,"html.parser")
    data = soup.findAll("li",{"class":"col-xs-6 col-sm-4 col-md-3 col-lg-3"})
#     print(data)
    for i in data:
        for j in i.findAll("a"):
            print("http://books.toscrape.com/"+j.get("href"))
            burl = "http://books.toscrape.com/"+j.get("href")
            break
#         for k in i.findAll("h3"):
#             print(k.text)
#             break
        req1 = requests.get(burl).content
        soup1 = BeautifulSoup(req1,"html.parser")
        data2 = soup1.findAll("div",{"class":"col-sm-6 product_main"})
        for i in data2:
            for j in i.findAll("h1"):
                print(j.text.strip())
                
        data2 = soup1.findAll("div",{"class":"item active"})
        for l in data2:
            for m in l.findAll("img"):
                print("http://books.toscrape.com/"+m.get("src"))
#                 print("-"*25)
        data3 = soup1.findAll("p")
        for i in data3:
            print(i.text.strip())
        print("-"*25)
except Exception as e:
    print("error occured :",e)
 ```
 
  ```python
try:
    import pyforest
    import requests
    from bs4 import BeautifulSoup
    import warnings
    warnings.filterwarnings("ignore")
    url = "https://timesofindia.indiatimes.com/topic/category-wise/news"
    intrest = input("enter ur category")
    url = f"https://timesofindia.indiatimes.com/topic/{intrest}"
    req = requests.get(url)
#     print(req)
    soup = BeautifulSoup(req.content,"html.parser")
    data = soup.findAll("div",{"class":"Mc7GB"})
    D = {}
    for pk,i in enumerate(data):
        d = {}
        for j in i.findAll("div",{"class":"EW1Mb _3v379"}):
#             print("Title :",j.text.strip())
            d["title"] = j.text.strip()
        for k in i.findAll("a"):
#             print("URL :",k.get("href"))
            d["URL"] = k.get("href")
        for l in i.findAll("div",{"class":"_13Z9I"}):
            for m in l.findAll("img"):
#                 print("Image URL :",m.get("src"))
                d["Image URL"] = m.get("src")
        for n in i.findAll("p",{'class':'itdvH _3v379'}):
#             print("Description :",n.text.strip())
            d["Description"] = n.text.strip()
        for o in i.findAll("div",{"class":"hVLK8"}):
#             s = o.text.strip()
#             s = s.split("/")[1].strip()
            d['Time'] = o.text.strip()
#             del s
#             print("-"*160) 
        D[pk] = d
        del d
    print("sample of DataSet :")
    print("-"*165)
    display(pd.DataFrame(D).T.head(2))
    print(f"{intrest} data fetched successfully.")
    print("-"*165)
except Exception as e:
    print("error occured :",e)
 ```

 ```python
try:
  import pyforest
  import requests
  from bs4 import BeautifulSoup
  import warnings
  warnings.filterwarnings("ignore")
  url = "https://timesofindia.indiatimes.com/topic/category-wise/news"
#   intrest = input("enter ur category")
#   url = f"https://timesofindia.indiatimes.com/topic/{intrest}"
  req = requests.get(url)
#     print(req)
  soup = BeautifulSoup(req.content,"html.parser")
  data = soup.findAll("div",{"class":"Mc7GB"})
  D = {}
  for pk,i in enumerate(data):
      d = {}
      for j in i.findAll("div",{"class":"EW1Mb _3v379"}):
#             print("Title :",j.text.strip())
          d["title"] = j.text.strip()
      for k in i.findAll("a"):
#             print("URL :",k.get("href"))
          d["URL"] = k.get("href")
          turl = k.get("href")
          req1 = requests.get(turl).content
          soup1 = BeautifulSoup(req1,"html.parser")
          for i2 in soup1.findAll("div",{"class":"_3YYSt clearfix"}):
                d["longDescription"] = i2.text.strip()
#               print(i.text.strip())
      for l in i.findAll("div",{"class":"_13Z9I"}):
          for m in l.findAll("img"):
#                 print("Image URL :",m.get("src"))
              d["Image URL"] = m.get("src")
      for n in i.findAll("p",{'class':'itdvH _3v379'}):
#             print("Description :",n.text.strip())
          d["Description"] = n.text.strip()
      for o in i.findAll("div",{"class":"hVLK8"}):
#             s = o.text.strip()
#             s = s.split("/")[1].strip()
          d['Time'] = o.text.strip()
#             del s
#             print("-"*160) 
      D[pk] = d
      del d
      
  print("sample of DataSet :")
  print("-"*165)
  display(pd.DataFrame(D).T.head(2))
  
  print(f"{intrest} data fetched successfully.")
  print("-"*165)
except Exception as e:
  print("error occured :",e)
  ```



```python
# download img from a web

try:
    
    import requests
    from bs4 import BeautifulSoup
    for i in range(1,10):
        url = f"https://www.indiatoday.in/science?page={i}"
        req = requests.get(url).content
        soup = BeautifulSoup(req,"html.parser")
        data = soup.findAll("div",{"class":"catagory-listing"})
        for i in data:
            for j in i.findAll('div',{"class":"pic"}):
                for k in j.findAll("img"):
                    print(k.get("src"))
        print()


#     print(soup.prettify())
#     print(req)
except Exception as e:
    print(e)
```
