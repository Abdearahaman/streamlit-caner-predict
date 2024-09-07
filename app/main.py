import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
   data = pd.read_csv("data/data.csv")
   
   
   data = data.drop(['Unnamed: 32', 'id'], axis=1)
   
   data['diagnosis'] = data.diagnosis.map({'M': 1, 'B': 0})
   
   
   return data



def add_sidebar():
      
    st.sidebar.header('Cell Nuclei Measurments')

    data = get_clean_data()


    slider_labels = [
    ('Radius (Mean)', 'radius_mean'),
    ('Texture (Mean)', 'texture_mean'),
    ('Perimeter (Mean)', 'perimeter_mean'),
    ('Area (Mean)', 'area_mean'),
    ('Smoothness (Mean)', 'smoothness_mean'),
    ('Compactness (Mean)', 'compactness_mean'),
    ('Concavity (Mean)', 'concavity_mean'),
    ('Concave Points (Mean)', 'concave points_mean'),
    ('Symmetry (Mean)', 'symmetry_mean'),
    ('Fractal Dimension (Mean)', 'fractal_dimension_mean'),
    ('Radius (Se)', 'radius_se'),
    ('Texture (Se)', 'texture_se'),
    ('Perimeter (Se)', 'perimeter_se'),
    ('Area (Se)', 'area_se'),
    ('Smoothness (Se)', 'smoothness_se'),
    ('Compactness (Se)', 'compactness_se'),
    ('Concavity (Se)', 'concavity_se'),
    ('Concave Points (Se)', 'concave points_se'),
    ('Symmetry (Se)', 'symmetry_se'),
    ('Fractal Dimension (Se)', 'fractal_dimension_se'),
    ('Radius (Worst)', 'radius_worst'),
    ('Texture (Worst)', 'texture_worst'),
    ('Perimeter (Worst)', 'perimeter_worst'),
    ('Area (Worst)', 'area_worst'),
    ('Smoothness (Worst)', 'smoothness_worst'),
    ('Compactness (Worst)', 'compactness_worst'),
    ('Concavity (Worst)', 'concavity_worst'),
    ('Concave Points (Worst)', 'concave points_worst'),
    ('Symmetry (Worst)', 'symmetry_worst'),
    ('Fractal Dimension (Worst)', 'fractal_dimension_worst')
]
 
    input_dict = {}
 
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
            step=0.01,
            format="%.2f"
        )
        
    return input_dict
 
 
 
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    
    input_array = np.array(list(input_data.values())).reshape(1,-1)
     
    input_array_scaled = scaler.transform(input_array) 
 
    prediction = model.predict(input_array_scaled)
     
    st.subheader("Cell Cluster prediction")
    st.write("The cell clluster is:")
     
    if prediction[0] == 0:
        st.write('<span class="diagnosis benign">Benign</span>',unsafe_allow_html=True)
    else:    
        st.write('<span class="diagnosis malicious">Malicious</span>',unsafe_allow_html=True)    
    
    st.write("Probbility of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probbility of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write('This app can assist medical professionals in making a diagnosis, but should not be used as a substitue for a professional diagnosis')
 
def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(["diagnosis"],axis=1) 
    
    scaled_dict = {}
    
    for key , value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min() 
        scaled_value = (value - min_value) / (max_value -min_value) 
        scaled_dict[key] = scaled_value
    
    return scaled_dict
        
        
def get_radar_chart(input_data):
    
    input_data =get_scaled_values(input_data) 
    
    categories = ['Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area'
                  , 'Smoothness','Compactness', 'Concavity', 'Concave points', 'Symmetry', 
                 'Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
              input_data['radius_mean'],
              input_data['texture_mean'],
              input_data['perimeter_mean'],
              input_data['area_mean'],
              input_data['smoothness_mean'],
              input_data['compactness_mean'],
              input_data['concavity_mean'],
              input_data['concave points_mean'],
              input_data['symmetry_mean'],
              input_data['fractal_dimension_mean']

          ],
      theta=categories,
      fill='toself',
      name='Mean value'
))
    fig.add_trace(go.Scatterpolar(
      r=[
              input_data['radius_se'],
              input_data['texture_se'],
              input_data['perimeter_se'],
              input_data['area_se'],
              input_data['smoothness_se'],
              input_data['compactness_se'],
              input_data['concavity_se'],
              input_data['concave points_se'],
              input_data['symmetry_se'],
              input_data['fractal_dimension_se']

          ],
      theta=categories,
      fill='toself',
      name='Standard error'
))
    fig.add_trace(go.Scatterpolar(
      r=[
              input_data['radius_worst'],
              input_data['texture_worst'],
              input_data['perimeter_worst'],
              input_data['area_worst'],
              input_data['smoothness_worst'],
              input_data['compactness_worst'],
              input_data['concavity_worst'],
              input_data['concave points_worst'],
              input_data['symmetry_worst'],
              input_data['fractal_dimension_worst']

          ],
      theta=categories,
      fill='toself',
      name='Worst value'
))

    fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
)
    return fig
    


def main():
    st.set_page_config(
        page_title="Breast Cancer detection",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    
    with st.container():
        st.title("Breast Cancer detector")
        st.write("Please connect this app to your cytology lab to help you diagnose breast cancer form your tissue sample .This app predicts using a machine learning model wether a breast mass is benign or malignant based on the measurments it receives from your cytosis lab.You can also update the measurments by hand using the sliders in the sidebar.")
    
    
    col1, col2 = st.columns([4,1]) 
       
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col2:
        add_predictions(input_data)   
             
        
if __name__ == "__main__":
    main()