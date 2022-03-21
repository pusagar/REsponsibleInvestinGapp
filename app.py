import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from prediction import get_prediction, ordinal_encoder,labelencoder

model = joblib.load(r'model/logreg.pkl')

st.set_page_config(page_title="RESponsible InvestinG App",
                   page_icon="🚧", layout="wide")


   

features = ['Yield (%)', 'Portfolio Sustainability Score', 'Portfolio Environmental Score', 
            'Portfolio Social Score', 'Portfolio Governance Score', '% Alcohol', '% Fossil Fuels', 
            '% Small Arms', '% Thermal Coal', '% Tobacco', '3 Years Annualized (%)', 
            '5 Years Annualized (%)', '10 Years Annualized (%)']

st.markdown("<h1 style='text-align: center;'>RESponsible InvestinG  App 🚧</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        
        yld = st.slider(" Yield (%)", 0.0, 34.08, value=0., step = 1., format="%f")
        psus = st.slider("Select Portfolio Sustainability Score: ", 10.38, 35.20,value=0., step = 1., format="%f")
        pes = st.slider("Select Portfolio Environmental Score", 0.0, 14.04, value=0., step = 1., format="%f")
        psos = st.slider("Select Portfolio Social Score", 0.0, 14.53, value=0., step = 1., format="%f")
        pgs = st.slider("Select Portfolio Governance Score", 0.0, 11.20, value=0., step = 1., format="%f")
        alcohol = st.slider("Select % Alcohol", 0.0, 19.97, value=0., step = 1., format="%f")
        fossil = st.slider("Select % Fossil Fuels", 0.0, 100.69, value=0., step = 1., format="%f")
        arms = st.slider("Select % Small Arms", 0.0, 7.91, value=0., step = 1., format="%f")
        coal = st.slider("Select % Thermal Coal", 0.0, 63.91, value=0., step = 1., format="%f")
        tobacco = st.slider("Select  % Tobacco", 0.0, 12.25, value=0., step = 1., format="%f")
        yrthree = st.slider("Select 3 Years Annualized (%)", -11.17, 49.75, value=-11.17, step = 1., format="%f")
        yrfive = st.slider("Select 5 Years Annualized (%)", -9.34, 36.51, value=-9.34, step = 1., format="%f")
        yrten = st.slider("Select 10 Years Annualized (%)", -5.86, 25.53, value=-5.86, step = 1., format="%f")
        
        
        submit = st.form_submit_button("Predict")


    if submit:

        data = np.array([yld,psus,pes,psos,pgs,alcohol,fossil,arms,coal,
                         tobacco,yrthree,yrfive,yrten]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)


        st.write(f"The class  is:  {pred[0]}")


if __name__ == '__main__':
    main()