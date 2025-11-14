import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

DIRETORIO_DESTINO_RAIZ = 'plots_pre'
DIRETORIO_DESTINO_HIST = 'histogramas'
DIRETORIO_DESTINO_BOX = 'boxplot'
DIRETORIO_DESTINO_RES = 'resultados'

DATASET = 'earthquake_data_tsunami.csv'

# -------------------- Criação dos Diretórios para os plots EDA --------------------
if not os.path.exists(DIRETORIO_DESTINO_RAIZ):
    os.makedirs(DIRETORIO_DESTINO_RAIZ)

caminho_histogramas = os.path.join(DIRETORIO_DESTINO_RAIZ, DIRETORIO_DESTINO_HIST)
if not os.path.exists(caminho_histogramas):
    os.makedirs(caminho_histogramas)

caminho_boxplot = os.path.join(DIRETORIO_DESTINO_RAIZ, DIRETORIO_DESTINO_BOX)
if not os.path.exists(caminho_boxplot):
    os.makedirs(caminho_boxplot)

if not os.path.exists(DIRETORIO_DESTINO_RES):
    os.makedirs(DIRETORIO_DESTINO_RES)

# -------------------- Definições para os Plots --------------------

titulos_colunas = { # Nomes para os títulos dos gráficos
    'magnitude': 'Magnitude do Terremoto', # Escala Richter (Preditor Primário)
    'cdi': 'Medida de Impacto na População', # Community Decimal Intensity
    'mmi': 'Indicador de Danos Estruturais', # Modified Mercalli Intensity
    'sig' : 'Significância do Evento', # Event Significance Score/Medição Geral de Perigo
    'depth' : 'Profundidade do Ponto Focal do Terremoto', # Mais próximo da superfície = pior o terremoto
    'latitude' : 'Latitude do Epicentro', # Indicador de Proximidade ao Oceano 
    'longitude' : 'Longitude do Epicentro', # Indicador de Proximidade ao Oceano
}

unidades_colunas = { # Unidades para os títulos dos gráficos
    'magnitude': 'escala richter',
    'cdi': 'CDI (escala de 0 a 9)',
    'mmi': 'MMI (escala de 0 a 12)',
    'sig' : '',
    'depth' : 'km',
    'latitude' : 'graus',
    'longitude' : 'graus',
}

# -------------------- Carregamento dos Dados / Divisão dos Dados / Normalização --------------------
df_original = pd.read_csv(DATASET)

remover_colunas = ['nst', 'dmin', 'gap', 'Year', 'Month']
df = df_original.drop(columns=remover_colunas, axis=1)

print("Realizando Divisão de Dados e Normalização")

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divisão de dados usado como exemplo para gerar os plots
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# -------------------- Plots com Atributos Mais Importantes e Valores Normalizados --------------------

print ("Gerando Plots com Dados Normalizados")

df_normalizado = pd.DataFrame(x_train, columns=x.columns)
colunas_plt = x.columns


for coluna in colunas_plt:
    titulo_grafico = titulos_colunas.get(coluna, coluna.capitalize())
    unidade = unidades_colunas.get(coluna, '')

    plt.figure(figsize=(9, 6))
    ax = df_normalizado[coluna].hist(bins=20, grid=False)  # ax é o objeto axes criado pelo pandas
    ax.set_title(f'Distribuição do Atributo: {titulo_grafico}')

    xlabel = titulo_grafico
    if unidade:
        xlabel = f'{unidade} (normalizado)'
        ax.set_xlabel(xlabel)
    
    ax.set_ylabel('Frequência')

    nome_base = coluna.lower().replace(' ', '_')
    nome_arquivo = f'{nome_base}_norma_hist.png'
    caminho_completo = os.path.join(caminho_histogramas, nome_arquivo)
    
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    plt.close()

axes = df_normalizado.hist(alpha=0.7, bins=20, figsize=(20, 8), layout=(3, 3), grid=False, sharex=False, sharey=False)

for ax in axes.flatten():
    ax.set_ylabel('Frequência', fontsize=10)
    col_name = ax.get_title()  # capture original column name before we change the title
    ax.set_xlabel(unidades_colunas.get(col_name, ''), fontsize=10)
    titulo_grafico = titulos_colunas.get(col_name)
    ax.set_title(titulo_grafico)

plt.tight_layout()

nome_arquivo = 'histograma_geral_normalizado.png' 
caminho_completo = os.path.join(caminho_histogramas, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

df_normalizado.boxplot(figsize=(12, 6))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

nome_arquivo = 'boxplot_geral_normalizado.png'
caminho_completo = os.path.join(caminho_boxplot, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

# -------------------- Treinamento dos Modelos, Predição e Resultados --------------------

print ("Realizando Treinamento, Predição e Resultados")

repetir = 10 # Número de repetições
base_seed = 50 # Valor base para reprodutibilidade dos testes
modelos = { # Modelos testados
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Redes Neurais": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
}
resultados_acuracia = {nome: [] for nome in modelos.keys()} # Para guardar resultados da acurácia

for i in range(1, repetir + 1):
    print("Realizando itereção número:", i)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=(base_seed + i), stratify=y)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for nome, modelo in modelos.items():
        nome_subdir = nome.replace(' ', '_').lower().replace('(', '').replace(')', '')
        caminho_sub_modelo = os.path.join(DIRETORIO_DESTINO_RES, nome_subdir)
        if not os.path.exists(caminho_sub_modelo):
            os.makedirs(caminho_sub_modelo)

        modelo.fit(x_train, y_train)
        y_pred = modelo.predict(x_test)
        results = classification_report(y_test, y_pred, digits=4)

        nome_arquivo = f"relatorio_{nome.replace(' ', '_').lower()}_{i}.txt"
        caminho_completo_relatorio = os.path.join(caminho_sub_modelo, nome_arquivo)
        
        with open(caminho_completo_relatorio, 'w') as f:
            f.write(results)

        # Guardando Dados para Análise Detalhada da Acurácia (Métrica Principal)
        dici_results = classification_report(y_test, y_pred, output_dict=True)
        acuracia = dici_results['accuracy']
        resultados_acuracia[nome].append(acuracia)

# -------------------- Resultados e Plot da Acurácia --------------------

print("Coletando Dados da Acurácia")

df_resultados = pd.DataFrame(resultados_acuracia)
df_resultados.to_csv(os.path.join(DIRETORIO_DESTINO_RES,'resultados_spot_checking_acc.csv'), index=False)

# Média e Desvio Padrão por Modelo
media_por_modelo = df_resultados.mean()
desvio_padrao_por_modelo = df_resultados.std()

nome_arquivo = f"relatorio_acc_media_desvio.txt"
caminho_completo_relatorio = os.path.join(DIRETORIO_DESTINO_RES, nome_arquivo)
        
with open(caminho_completo_relatorio, 'w') as f:
    f.write("Média por Modelo:\n")
    f.write(media_por_modelo.to_string())
    f.write("\n\nDesvio Padrão por Modelo:\n")
    f.write(desvio_padrao_por_modelo.to_string())

# Boxplot
df_resultados.boxplot(figsize=(12, 6))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

nome_arquivo = 'boxplot_acc.png'
caminho_completo = os.path.join(DIRETORIO_DESTINO_RES, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

print("Spot-checking Finalizado")