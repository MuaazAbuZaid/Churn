import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

def prediction(Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    test_df = pd.DataFrame(columns= Inputs)
    test_df.at[0,"Geography"] = Geography
    test_df.at[0,"CreditScore"] = CreditScore
    test_df.at[0,"Gender"] = Gender
    test_df.at[0,"Age"] = Age
    test_df.at[0,"Tenure"] = Tenure
    test_df.at[0,"Balance"] = Balance
    test_df.at[0,"NumOfProducts"] = NumOfProducts
    test_df.at[0,"HasCrCard"] = HasCrCard
    test_df.at[0,"IsActiveMember"] = IsActiveMember
    test_df.at[0,"EstimatedSalary"] = EstimatedSalary
    
    result = Model.predict_proba(test_df)[0][1]

    if result >= 0.3:
        result = 1
    else:
        result = 0    
    return result

def main():
    st.title("Churn Prediction")
    Geography = st.selectbox("Geography" , ['Germany', 'Spain', 'France'])
    Gender = st.selectbox("Gender" , ['Male', 'Female'])
    CreditScore = st.slider("CreditScore" , min_value= 0 , max_value=1000 , value=0,step=10) 
    Age = st.slider("Age" , min_value= 10 , max_value=100 , value=10,step=1)   
    Tenure = st.slider("Tenure" , min_value= 0 , max_value=10 , value=0,step=1) 
    Balance = st.slider("Balance" , min_value= 100 , max_value=500000 , value=100,step=100) 
    NumOfProducts = st.slider("NumOfProducts" , min_value= 1 , max_value=5 , value=1,step=1)
    HasCrCard = st.slider("HasCrCard" , min_value= 0 , max_value=1 , value=0,step=1)
    IsActiveMember = st.slider("IsActiveMember" , min_value= 0 , max_value=1 , value=0,step=1)
    EstimatedSalary = st.slider("EstimatedSalary" , min_value= 10 , max_value=200000 , value=10,step=10)  

    if st.button("predict"):
        result = prediction(Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        outp = ['safe', 'risk']
        st.text(f"It's {outp[result]}")

if __name__ == '__main__':
    main()
