import pandas as pd

train_data = pd.read_csv("train.csv")

#print(train_data.head())

#print(train_data.describe())

#sprint(train_data["Transported"].value_counts())
print(train_data["Destination"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns 
# correlation = train_data.corr()
# heatmap = sns.heatmap(correlation,annot=True)

# heatmap.set(xlabel="Features", ylabel="Features")
# plt.show()


# print(train_data.shape)
print(train_data.dtypes)

train_data["Transported"] = train_data["Transported"].map({True:1,False:0})

y = train_data["Transported"]

x= train_data.drop(["Transported"],axis=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

cols = x.select_dtypes(include=['float64']).columns
sc_data = scaler.fit_transform(x[cols])

sc_dataframe1 = pd.DataFrame(sc_data,columns=cols)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

cat_data1 = x.select_dtypes(include=['object']).copy()
en_data = cat_data1.apply(encoder.fit_transform)

X = pd.concat([sc_dataframe1,en_data],axis=1)

# print(X.shape)
# print(X.head())
# print(y.head())
print(X.isnull().sum())

X["Age"].fillna(X["Age"].mean(),inplace=True)
X["RoomService"].fillna(X["RoomService"].mean(),inplace=True)
X["FoodCourt"].fillna(X["FoodCourt"].mean(),inplace=True)
X["ShoppingMall"].fillna(X["ShoppingMall"].mean(),inplace=True)
X["Spa"].fillna(X["Spa"].mean(),inplace=True)
X["VRDeck"].fillna(X["VRDeck"].mean(),inplace=True)

print(X.isnull().sum())


import pickle 

pk = open("X.pickle","wb")
pickle.dump(X,pk)
pk.close()

pk = open("y.pickle","wb")
pickle.dump(y,pk)
pk.close()











