import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIRETORIO_DESTINO_RAIZ = 'plots_eda'
DIRETORIO_DESTINO_HIST = 'histogramas'
DIRETORIO_DESTINO_BOX = 'boxplot'
DIRETORIO_DESTINO_ALVO = 'alvo'
DATASET = 'earthquake_data_tsunami.csv'

# Carregamento dos Dados
df = pd.read_csv(DATASET)

# Criação dos Diretórios para os plots EDA
if not os.path.exists(DIRETORIO_DESTINO_RAIZ):
    os.makedirs(DIRETORIO_DESTINO_RAIZ)

caminho_histogramas = os.path.join(DIRETORIO_DESTINO_RAIZ, DIRETORIO_DESTINO_HIST)
if not os.path.exists(caminho_histogramas):
    os.makedirs(caminho_histogramas)

caminho_boxplot = os.path.join(DIRETORIO_DESTINO_RAIZ, DIRETORIO_DESTINO_BOX)
if not os.path.exists(caminho_boxplot):
    os.makedirs(caminho_boxplot)

caminho_alvo = os.path.join(DIRETORIO_DESTINO_RAIZ, DIRETORIO_DESTINO_ALVO)
if not os.path.exists(caminho_alvo):
    os.makedirs(caminho_alvo)

# -------------------- Definições para os Plots --------------------

titulos_colunas = { # Nomes para os títulos dos gráficos
    'magnitude': 'Magnitude do Terremoto', # Escala Richter (Preditor Primário)
    'cdi': 'Medida de Impacto na População', # Community Decimal Intensity
    'mmi': 'Indicador de Danos Estruturais', # Modified Mercalli Intensity
    'sig' : 'Significância do Evento', # Event Significance Score/Medição Geral de Perigo
    'nst' : 'Número de Estações de Monitoramento Sísmico', # Apenas para indicação de qualidade dos dados
    'dmin' : 'Distância para a Estação de Monitoramento Mais Próxima', # Apenas para indicação de qualidade dos dados
    'gap' : 'Distância Azimutal Entre Estações', # Confiabilidade da Localização (qualidade de dados)
    'depth' : 'Profundidade do Ponto Focal do Terremoto', # Mais próximo da superfície = pior o terremoto
    'latitude' : 'Latitude do Epicentro', # Indicador de Proximidade ao Oceano 
    'longitude' : 'Longitude do Epicentro', # Indicador de Proximidade ao Oceano
    'Year' : 'Ano', # Padrões Temporais
    'Month' : 'Mês', # Análise Sazonal
}

unidades_colunas = { # Unidades para os títulos dos gráficos
    'magnitude': 'escala richter',
    'cdi': 'CDI (escala de 0 a 9)',
    'mmi': 'MMI (escala de 0 a 12)',
    'sig' : '',
    'nst' : 'número de estações',
    'dmin' : 'km',
    'gap' : 'graus',
    'depth' : 'km',
    'latitude' : 'graus',
    'longitude' : 'graus',
    'Year' : '',
    'Month' : '',
}

print ("Gerando Plots EDA")

# -------------------- Plots dos Histogramas para cada Atributo --------------------

df_features = df.iloc[:, :-1] # Remover atributo alvo para os plots
colunas_plt = df_features.columns

for coluna in colunas_plt:
    titulo_grafico = titulos_colunas.get(coluna, coluna.capitalize())
    unidade = unidades_colunas.get(coluna, '')

    plt.figure(figsize=(9, 6))
    ax = df[coluna].hist(bins=20, grid=False)  # ax é o objeto axes criado pelo pandas
    ax.set_title(f'Distribuição do Atributo: {titulo_grafico}')

    xlabel = titulo_grafico
    if unidade:
        xlabel = f'{unidade}'
        ax.set_xlabel(xlabel)
    
    ax.set_ylabel('Frequência')

    nome_base = coluna.lower().replace(' ', '_')
    nome_arquivo = f'{nome_base}_hist.png'
    caminho_completo = os.path.join(caminho_histogramas, nome_arquivo)
    
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------- Plot do Histograma Geral --------------------

axes = df.drop(columns=['tsunami']).hist(alpha=0.7, bins=20, figsize=(20, 8), layout=(3, 4), grid=False, sharex=False, sharey=False)

for ax in axes.flatten():
    ax.set_ylabel('Frequência', fontsize=10)
    x_label = unidades_colunas.get(ax.get_title())
    ax.set_xlabel(f'{x_label}', fontsize=10)
    titulo_grafico = titulos_colunas.get(ax.get_title())
    ax.set_title(titulo_grafico)

plt.tight_layout()

nome_arquivo = 'histograma_geral.png' 
caminho_completo = os.path.join(caminho_histogramas, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

# -------------------- Plot do Boxplot Geral --------------------

df.drop(columns=['tsunami']).rename(columns=str.upper).boxplot(figsize=(12, 6))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

nome_arquivo = 'boxplot_geral.png'
caminho_completo = os.path.join(caminho_boxplot, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

# -------------------- Plot da Distribuição do Atributo Alvo --------------------

target_col = df.columns[-1] # Só coluna alvo
class_counts = df[target_col].value_counts() # Quantos elementos tem o atributo alvo

plt.figure(figsize=(7, 5))
class_counts.plot(kind='bar')

plt.title(f'Distribuição de Classes da Coluna Alvo')
plt.xlabel('Classe')
plt.ylabel('Número de Instrâncias')

for i, v in enumerate(class_counts): # Coloca os número em cima das barras
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=0)
plt.tight_layout()

nome_arquivo = 'distribuicao_alvo.png'
caminho_completo = os.path.join(caminho_alvo, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)