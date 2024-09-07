
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data.diagnosis
    
    # Scaling Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data    
    X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=  0.2, random_state=42)
        
    model = XGBClassifier(
       random_state=42,
       eta=0.05,
       max_depth=5,
       n_estimators=120,
       gamma=0.01,
       subsample=0.2,
       colsample_bytree=0.4,
       reg_alpha=0.6,
       reg_lambda=0.3
    )
    model.fit(X_train, y_train)
    
    
    y_pred =model.predict(X_test)
   
    print(data.describe())
    print('Accuracy of our model', accuracy_score(y_test, y_pred))
    print('Classification report', classification_report( y_test, y_pred)) 
    return model, scaler



   
    
    
    
    
    
def get_clean_data():
   data = pd.read_csv("data/data.csv")
   
   
   data = data.drop(['Unnamed: 32', 'id'], axis=1)
   
   data['diagnosis'] = data.diagnosis.map({'M': 1, 'B': 0})
   
   
   return data



def main(): 
   data = get_clean_data()
   model, scaler = create_model(data)
   
   with open('model/model.pkl','wb') as f:
      pickle.dump(model, f) 
      
   with open('model/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f) 
    
    
if __name__ == '__main__':
    main()