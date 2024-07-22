import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#criação de titulo com streamlit
st.title("Previsão de custo inicial para franquia")

#lendo csv
data = pd.read_csv('Projeto 1/slr12.csv', sep=';')

#X maiusculo e y minusculo
X = data[['FrqAnual']] #dois colchetes para ser dataframe
y = data['CusInic'] #serie do pandas

#fit() é usado para treinar o modelo 
modelo = LinearRegression().fit(X,y)

st.header('Valor anual da franquia')
novo_valor = st.number_input("Insira novo valor", min_value=1.0, max_value=99999999.0, value=1500.0, step=10.0)
processar = st.button('Processar')

if processar:
    dados_novo_valor = pd.DataFrame([[novo_valor]], columns=['FrqAnual'])
    prev = modelo.predict(dados_novo_valor)
    st.header(f"Previsão de custo inicial R$: {prev[0]:.2f}")
    
#criando colunas na visualização do streamlit
col1, col2 = st.columns(2)

with col1:
    st.header('Dados') #criação de cabeçalho para coluna com streamlit
    st.table(data.head(10)) #criação de tabela estatica com streamlit

with col2:
    st.header("Gráfico de Dispersão")
    fig, ax = plt.subplots() #criação de figura e eixo para o grafico
    ax.scatter(X, y, color = 'blue') #plota grafico de disperção (scatter plot)
    ax.plot(X, modelo.predict(X), color='red') #plota linha com a previsão do modelo
    
    st.pyplot(fig) #Exibe a figura criada no Streamlit

