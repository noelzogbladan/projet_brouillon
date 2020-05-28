import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as smf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
#import plotly.express as px
import seaborn as sns
import altair as alt 
import os
import time
import random
st.markdown("<h1 style='text-align: center; color: red;'>K-flix</h1>", unsafe_allow_html=True)
st.image("image.png", width=800)
df_victory1=pd.read_csv("https://raw.githubusercontent.com/noelzogbladan/noelzogbladan/master/df_victory1.csv")
df_victory=pd.read_csv("https://raw.githubusercontent.com/noelzogbladan/noelzogbladan/master/df_victory.csv")

st.markdown("<h1 style='text-align: center; color:black;'>Système de recommandation</h1>", unsafe_allow_html=True)
 
if st.sidebar.checkbox("Recherche par genre"): 
        def get_film_per_genre():
                col_genre = ['Autres', 'Action', 'Adventure',
                'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']  
                result = 0
                selection=st.selectbox("Selectionner un genre de film",col_genre)
                for i in col_genre:
                        if selection == i:
                                result = df_victory1[df_victory1[i]==1][['title',i]].groupby('title').mean()
                                result = result.reset_index()
                                result = result['title'].tolist()  
                                print(random.sample(result, 5))
                        
                                return random.sample(result,5)
        selected_films=get_film_per_genre()
        st.info('Cool voici les films que tu vas adorer regarder {}'.format(selected_films))
                #filename=file_selector()
                #get_film_per_genre(choice= input("Selectionner un genre :")))
if st.sidebar.checkbox("Recherche par lettres"): 
        partial=st.text_input("Ecris les premières lettres du film recherché")
        all_film_names=list(df_victory1.title.values)
        def get_id_from_partial_name(partial):
                for name in all_film_names:
                        if partial in name:
                                print(name)
                                return name
            #get_id_from_partial_name(partial)
        films_recherches=get_id_from_partial_name(partial)
            
        st.info('Voici les films {}'.format(films_recherches))
    # if __name__ =='__main__':
            # main()

#def main():

if st.sidebar.checkbox("Recommandation de films"): 
    @st.cache(allow_output_mutation=True)
    def model():
        X = df_victory.drop(['title','movieId','year'], axis=1)
        y = df_victory['title']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        modelKNN = KNeighborsClassifier().fit(X_train_scaled, y_train)
        modelKNN.predict(X_test)
        return modelKNN 
               
    titles = list(df_victory1.title.values)
    name = st.selectbox("Choisis un film",titles)
        #name = st.text_input("")
    f = df_victory[df_victory["title"]== name].index[0]
    film = df_victory[df_victory.index == f]
    voisin1 = model().kneighbors(film.loc[:,['year', 'Autres', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery','Romance',
                                          'Sci-Fi', 'Thriller', 'War', 'Western', 'rating']]) [1][0]
    for i in voisin1:
        neigh = pd.DataFrame(df_victory[df_victory.index == i]['title'])
        print(neigh)
        st.info(neigh['title'].tolist())
    st.balloons()
           #st.info(neigh)
        #st.info('Voici les films que je te recommande de regarder à coups sur {}'.format(films_recommended))
  