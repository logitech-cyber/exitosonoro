import streamlit as st
import pandas as pd
import re
import datetime
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import gradio as gr

warnings.filterwarnings("ignore")

def main():
    # Cargar los datos
    data_file = st.file_uploader("Selecciona un archivo CSV", type="csv")

    # Verificar si se ha subido un archivo
    if data_file is not None:
        # Leer el archivo CSV en Streamlit
        df_final = pd.read_csv(data_file)

        df_final['release'] = df_final['release'].replace('[^a-zA-Z0-9\s]', '', regex=True)
        df_final['date'] = df_final['date'].replace('[^a-zA-Z0-9\s]', '', regex=True)
        df_final['release'] = pd.to_datetime(df_final['release'], format='%Y%m%d')
        df_final['date'] = pd.to_datetime(df_final['date'], format='%Y%m%d')

        st.write(df_final)

        # División general de los datos
        le = LabelEncoder()
        df_final['name_codificado'] = le.fit_transform(df_final['name'])
        df_final['song_codificado'] = le.fit_transform(df_final['song'])
        df_final['genre_codificado'] = le.fit_transform(df_final['genre_song'])

        # Crear x (quitar la columna o campo de "popularity_song")
        X = df_final.drop('popularity_song', axis=1)

        # Crear y (Valor de la columna "popularity_song")
        y = df_final['popularity_song']

        # Dividir los datos para entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

        # Definir el modelo XGBoost
        model = XGBRegressor()

        # Pipeline de la predicción completa
        numeric_list = ['avg_vote', 'votes', 'rank', 'peak-rank', 'weeks-on-board', 'energy', 'loudness', 'danceability']
        categorical_list = ['song_codificado', 'name_codificado', 'genre_codificado']

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                  ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                                             unknown_value=np.nan))])

        transformer = ColumnTransformer([('num', numeric_transformer, numeric_list),
                                         ('cat', categorical_transformer, categorical_list)])

        model_pipeline = Pipeline([('transformer', transformer), ('model', model)])

        # Ajustar el modelo
        model_pipeline.fit(X_train, y_train)

        def predecir_popularidad(avg_vote, votes, rank, peak_rank, weeks_on_board, energy, loudness, song_name):
            # Verificar si la canción existe en los datos de entrenamiento
            if song_name in df_final['song'].values:
                # Obtener el valor de song_codificado correspondiente al nombre de la canción ingresada
                song_codificado = df_final.loc[df_final['song'] == song_name, 'song_codificado'].values[0]

                # Obtener el nombre de la película asociada a la canción
                movie_name = df_final.loc[df_final['song'] == song_name, 'name'].values[0]

                # Verificar si la película existe en los datos de entrenamiento
                if movie_name in df_final['name'].values:
                    # Obtener el valor de name_codificado correspondiente al nombre de la película
                    name_codificado = df_final.loc[df_final['name'] == movie_name, 'name_codificado'].values[0]
                else:
                    name_codificado = np.nan

                # Obtener el valor de genre_codificado correspondiente al nombre de la canción ingresada
                genre_codificado = df_final.loc[df_final['song'] == song_name, 'genre_codificado'].values[0]

                # Obtener el valor de date correspondiente al nombre de la canción ingresada
                date = df_final.loc[df_final['song'] == song_name, 'date'].values[0]

                # Obtener el valor de danceability correspondiente al nombre de la canción ingresada
                danceability = df_final.loc[df_final['song'] == song_name, 'danceability'].values[0]

                # Crear un DataFrame con los datos de entrada
                input_data = pd.DataFrame([[avg_vote, votes, rank, peak_rank, weeks_on_board, energy, loudness,
                                            song_codificado, name_codificado, genre_codificado, date, danceability]],
                                          columns=['avg_vote', 'votes', 'rank', 'peak-rank', 'weeks-on-board', 'energy',
                                                   'loudness', 'song_codificado', 'name_codificado',
                                                   'genre_codificado', 'date', 'danceability'])

                # Realizar la predicción utilizando el pipeline
                prediction = model_pipeline.predict(input_data)

                # Obtener las películas en las que ha aparecido la canción
                peliculas = df_final.loc[df_final['song_codificado'] == song_codificado, 'name'].unique()

                return prediction, song_name, peliculas
            else:
                return None, song_name, None

        def predict_interface(avg_vote, votes, rank, peak_rank, weeks_on_board, energy, loudness, song_name):
            prediction, song, peliculas = predecir_popularidad(avg_vote, votes, rank, peak_rank, weeks_on_board, energy, loudness, song_name)

            if prediction is not None:
                result = f"La canción {song} tendrá una popularidad de: {prediction}"
                if len(peliculas) > 0:
                    peliculas_str = "\n".join(peliculas)
                    result += "\ny ha sido utilizada en las siguientes películas:\n" + peliculas_str
                else:
                    result += "\nLa canción no ha aparecido en ninguna película."
            else:
                result = f"La canción {song} no existe en los datos de entrenamiento. Por favor, ingrese otra canción."

            return result

        # Interfaz de Gradio
        app = gr.Interface(fn=predict_interface,inputs=["number", "number", "number", "number", "number", "number", "number", "text"],outputs="text",title="Prediccion de popularidad (XGBOOST)", description="Ingrese los valores correspondientes para obtener la predicción de popularidad de una canción")

        app.launch()

if __name__ == "__main__":
    main()
