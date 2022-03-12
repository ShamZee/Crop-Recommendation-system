import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

# -----------------------------------------------Starting Point--------------------------------------------------
import streamlit as st

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


# ----------------sidebar Models ------------------
option = st.sidebar.selectbox(
         'Select any Model',
         ('Select Model', 'Suport Vector Machine', 'Logistic Regression','Decision Tree','RandomForest'))

#-----------------Dataset------------------------
dataset = pickle.load(open('data_set.pkl','rb'))
data = pd.DataFrame(dataset)

# ----------------Model Load---------------------
pickle_in1 = open('model.pkl',"rb")
model_logistic = pickle.load(pickle_in1)
pickle_in2 = open('model_Dt.pkl',"rb")
model_Dt = pickle.load(pickle_in2)
pickle_in3 = open('model_SVC.pkl',"rb")
model_SVC = pickle.load(pickle_in3)
pickle_in4 = open('model_rf.pkl',"rb")
model_rf = pickle.load(pickle_in4)

# -----------Main----------------
def main():
    html_temp = """ 
    <div style="background-color:tomato;padding:4px">
    <h2 style="color:white;text-align:center;">Prediction</h2>"""
    st.markdown(html_temp,unsafe_allow_html=True)


    # -----------------------------
    N = st.number_input(label='NITROGEN',min_value=0,max_value=300)
    P = st.number_input('PHOSPHOROUS', min_value=0,max_value=300)
    K = st.number_input('POTASSIUM', min_value=0,max_value=250)
    tem = st.number_input('TEMPERATURE', min_value=0,max_value=50)
    Humidity = st.number_input('HUMIDITY', min_value=0,max_value=250)
    Soil_acidity = st.number_input('SOIL ACIDITY', min_value=0,max_value=250)
    Rain = st.number_input('RAINFALL', min_value=0,max_value=300)

    #----------------------Button------------------------
    result=[] ;
    info = ""
    if st.button("PREDICT"):
        if option=="Logistic Regression":
            info = "Preictef By Logistic Regresion"
            result = predict_note_authentication(N,P,K,tem,Humidity,Soil_acidity,Rain)
            st.write('Predicted crop is : {}'.format(result[0]))

        elif option=="RandomForest":
            info = "Preictef By RandomForest"
            result = predict_By_rf(N,P,K,tem,Humidity,Soil_acidity,Rain)
            st.write('Predicted crop is : {}'.format(result[0]))
        elif option=="Suport Vector Machine":
            info = "Preictef By Suport vrctore Machine"
            result = predict_By_SVC(N,P,K,tem,Humidity,Soil_acidity,Rain)
            st.write('Predicted crop is : {}'.format(result[0]))

        elif option =="Decision Tree":
            info = "Preictef By Decision Tree"
            result = predict_By_DT(N,P,K,tem,Humidity,Soil_acidity,Rain)
            st.write('Predicted crop is : {}'.format(result[0]))



    st.write('Model : {}'.format(info))
    return result



def predict_note_authentication(N,P,K,tem,Humidity,Soil_acidity,Rain):
    prediction = model_logistic.predict([[N,P,K,tem,Humidity,Soil_acidity,Rain]])
    return prediction

def predict_By_SVC(N,P,K,tem,Humidity,Soil_acidity,Rain):
    prediction = model_SVC.predict([[N,P,K,tem,Humidity,Soil_acidity,Rain]])
    return prediction

def predict_By_DT(N,P,K,tem,Humidity,Soil_acidity,Rain):
    prediction = model_Dt.predict([[N,P,K,tem,Humidity,Soil_acidity,Rain]])
    return prediction

def predict_By_rf(N,P,K,tem,Humidity,Soil_acidity,Rain):
    prediction = model_rf.predict([[N,P,K,tem,Humidity,Soil_acidity,Rain]])
    return prediction

# ---------------------------Info------------------------------

col_1, col_2,col_3= st.columns((3,1,5))
with col_1:
    if __name__ == '__main__':
        pred = main()

with col_2:
    pass
with col_3:


    html_temp = """
        <div style="background-color:tomato;padding:4px">
        <h2 style="color:white;text-align:center;">Details</h2>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    def details(predited_crop):
        st.selectbox(
            'Detail About every Crop',
            predited_crop)
        # -------------------------------
        # -------------------------------------------
        if len(predited_crop)>0:
            x = data[data['label'] == predited_crop[0]]
            # -------------------------------------------
            dat = {'Minimum': [x['N'].min(),x['P'].min(),x['K'].min(),x['temperature'].min(),x['humidity'].min(),x['ph'].min(),x['rainfall'].min()],
                    'Average': [x['N'].mean(),x['P'].mean(),x['K'].mean(),x['temperature'].mean(),x['humidity'].mean(),x['ph'].mean(),x['rainfall'].mean()],
                    'Maximum':[x['N'].max(),x['P'].max(),x['K'].max(),x['temperature'].max(),x['humidity'].max(),x['ph'].max(),x['rainfall'].max()]}

            # Create DataFrame
            df = pd.DataFrame(dat)
            df = df.set_index(data.columns[:7])
            return df

    st.table(details(pred))

#---------------------------------------------------------End---------------------------------------------------------




