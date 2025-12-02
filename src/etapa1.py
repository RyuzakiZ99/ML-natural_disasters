import os
import math
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

# -------------------- Definições de Diretórios e Dataset --------------------
DIRETORIO_ETAPA1 = 'etapa_1'

DIRETORIO_DESTINO_RAIZ_EDA = 'plots_eda'
DIRETORIO_DESTINO_HIST_EDA = 'histogramas'
DIRETORIO_DESTINO_BOX_EDA = 'boxplot'
DIRETORIO_DESTINO_ALVO_EDA = 'alvo'

DIRETORIO_DESTINO_RAIZ_PRE = 'plots_pre'
DIRETORIO_DESTINO_HIST_PRE = 'histogramas'
DIRETORIO_DESTINO_BOX_PRE = 'boxplot'

DIRETORIO_DESTINO_RES_SPOT = 'resultados_spot'
DIRETORIO_DESTINO_ACC_SPOT = 'acurácia'
DIRETORIO_DESTINO_RECALL_SPOT = 'recall'
DIRETORIO_DESTINO_PREC_SPOT = 'precição'
DIRETORIO_DESTINO_ITE_MODELOS_SPOT = 'iterações_modelos'

DATASET = 'earthquake_data_tsunami.csv'

# -------------------- Criação dos Diretórios para os plots EDA --------------------

caminho_eda = os.path.join(DIRETORIO_ETAPA1, DIRETORIO_DESTINO_RAIZ_EDA)
if not os.path.exists(caminho_eda):
    os.makedirs(caminho_eda)

caminho_histogramas_eda = os.path.join(caminho_eda, DIRETORIO_DESTINO_HIST_EDA)
if not os.path.exists(caminho_histogramas_eda):
    os.makedirs(caminho_histogramas_eda)

caminho_boxplot_eda = os.path.join(caminho_eda, DIRETORIO_DESTINO_BOX_EDA)
if not os.path.exists(caminho_boxplot_eda):
    os.makedirs(caminho_boxplot_eda)

caminho_alvo_eda = os.path.join(caminho_eda, DIRETORIO_DESTINO_ALVO_EDA)
if not os.path.exists(caminho_alvo_eda):
    os.makedirs(caminho_alvo_eda)

# -------------------- Criação dos Diretórios para os plots Pré-Processamento --------------------

caminho_pre = os.path.join(DIRETORIO_ETAPA1, DIRETORIO_DESTINO_RAIZ_PRE)
if not os.path.exists(caminho_pre):
    os.makedirs(caminho_pre)

caminho_histogramas_pre = os.path.join(caminho_pre, DIRETORIO_DESTINO_HIST_PRE)
if not os.path.exists(caminho_histogramas_pre):
    os.makedirs(caminho_histogramas_pre)

caminho_boxplot_pre = os.path.join(caminho_pre, DIRETORIO_DESTINO_BOX_PRE)
if not os.path.exists(caminho_boxplot_pre):
    os.makedirs(caminho_boxplot_pre)

# -------------------- Criação dos Diretórios para os Resultados do Spot-checking --------------------

caminho_res = os.path.join(DIRETORIO_ETAPA1, DIRETORIO_DESTINO_RES_SPOT)
if not os.path.exists(caminho_res):
    os.makedirs(caminho_res)

caminho_iteracoes_modelos = os.path.join(caminho_res, DIRETORIO_DESTINO_ITE_MODELOS_SPOT)
if not os.path.exists(caminho_iteracoes_modelos):
    os.makedirs(caminho_iteracoes_modelos)

caminho_acuracia = os.path.join(caminho_res, DIRETORIO_DESTINO_ACC_SPOT)
if not os.path.exists(caminho_acuracia):
    os.makedirs(caminho_acuracia)

caminho_recall = os.path.join(caminho_res, DIRETORIO_DESTINO_RECALL_SPOT)
if not os.path.exists(caminho_recall):
    os.makedirs(caminho_recall)

caminho_precisao = os.path.join(caminho_res, DIRETORIO_DESTINO_PREC_SPOT)
if not os.path.exists(caminho_precisao):
    os.makedirs(caminho_precisao)

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

# -------------------- EDA --------------------
df = pd.read_csv(DATASET)
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
    caminho_completo = os.path.join(caminho_histogramas_eda, nome_arquivo)
    
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
caminho_completo = os.path.join(caminho_histogramas_eda, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

# -------------------- Plot do Boxplot Geral --------------------

df_boxplot_eda = df.drop(columns=['tsunami']).rename(columns=str)
fig, ax = plt.subplots(figsize=(12, 6))
df_boxplot_eda.boxplot(ax=ax)

ax.set_xticklabels(df_boxplot_eda.columns, rotation=45, ha='right')
plt.tight_layout()

nome_arquivo = 'boxplot_geral.png'
caminho_completo = os.path.join(caminho_boxplot_eda, nome_arquivo)
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
caminho_completo = os.path.join(caminho_alvo_eda, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)

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
    caminho_completo = os.path.join(caminho_histogramas_pre, nome_arquivo)
    
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
caminho_completo = os.path.join(caminho_histogramas_pre, nome_arquivo)
plt.savefig(caminho_completo, dpi=300)
plt.close()

# CORREÇÃO: Uso da API orientada a objetos do Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
df_normalizado.boxplot(ax=ax)

# Define os rótulos e aplica a rotação
ax.set_xticklabels(df_normalizado.columns, rotation=45, ha='right')

plt.tight_layout()

nome_arquivo = 'boxplot_geral_normalizado.png'
caminho_completo = os.path.join(caminho_boxplot_pre, nome_arquivo)
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
resultados_recall = {nome: [] for nome in modelos.keys()} # Para guardar resultados do recall
resultados_precision = {nome: [] for nome in modelos.keys()} # Para guardar resultados da precisão

for i in range(1, repetir + 1):
    print("Realizando itereção número:", i)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=(base_seed + i), stratify=y)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for nome, modelo in modelos.items():
        nome_subdir = nome.replace(' ', '_').lower().replace('(', '').replace(')', '')
        caminho_sub_modelo = os.path.join(caminho_iteracoes_modelos, nome_subdir)
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

        recall_pos = None
        if '1' in dici_results:
            recall_pos = dici_results['1']['recall']
        elif 1 in dici_results:  # no caso dos rótulos serem int keys em vez de str
            recall_pos = dici_results[1]['recall']
        else:
            recall_pos = dici_results.get('macro avg', {}).get('recall', None)

        precision_pos = None
        if '1' in dici_results:
            precision_pos = dici_results['1']['precision']
        elif 1 in dici_results:  # no caso dos rótulos serem int keys em vez de str
            precision_pos = dici_results[1]['precision']
        else:
            precision_pos = dici_results.get('macro avg', {}).get('precision', None)


        resultados_acuracia[nome].append(acuracia)
        resultados_recall[nome].append(recall_pos)
        resultados_precision[nome].append(precision_pos)
# -------------------- Resultados e Plot da Acurácia --------------------

print("Coletando Dados e Gerando Resultados Finais")

df_resultados_acc = pd.DataFrame(resultados_acuracia)
df_resultados_acc.to_csv(os.path.join(caminho_acuracia,'resultados_spot_checking_acc.csv'), index=False)

df_resultados_recall = pd.DataFrame(resultados_recall)
df_resultados_recall.to_csv(os.path.join(caminho_recall,'resultados_spot_checking_recall.csv'), index=False)

df_resultados_precision = pd.DataFrame(resultados_precision)
df_resultados_precision.to_csv(os.path.join(caminho_precisao,'resultados_spot_checking_precision.csv'), index=False)

# Média e Desvio Padrão por Modelo
media_por_modelo_acc = df_resultados_acc.mean()
desvio_padrao_por_modelo_acc = df_resultados_acc.std()

media_por_modelo_precision = df_resultados_precision.mean()
desvio_padrao_por_modelo_precision = df_resultados_precision.std()

nome_arquivo_acc = f"relatorio_acc_media_desvio.txt"
caminho_completo_relatorio = os.path.join(caminho_acuracia, nome_arquivo_acc)

nome_arquivo_recall = f"relatorio_recall_media_desvio.txt"
caminho_completo_relatorio_recall = os.path.join(caminho_recall, nome_arquivo_recall)

nome_arquivo_precision = f"relatorio_precision_media_desvio.txt"
caminho_completo_relatorio_precision = os.path.join(caminho_precisao, nome_arquivo_precision)
        
with open(caminho_completo_relatorio, 'w') as f:
    f.write("Média por Modelo:\n")
    f.write(media_por_modelo_acc.to_string())
    f.write("\n\nDesvio Padrão por Modelo:\n")
    f.write(desvio_padrao_por_modelo_acc.to_string())

with open(caminho_completo_relatorio_recall, 'w') as f:
    media_por_modelo_recall = df_resultados_recall.mean()
    desvio_padrao_por_modelo_recall = df_resultados_recall.std()
    f.write("Média por Modelo:\n")
    f.write(media_por_modelo_recall.to_string())
    f.write("\n\nDesvio Padrão por Modelo:\n")
    f.write(desvio_padrao_por_modelo_recall.to_string())

with open(caminho_completo_relatorio_precision, 'w') as f:
    f.write("Média por Modelo:\n")
    f.write(media_por_modelo_precision.to_string())
    f.write("\n\nDesvio Padrão por Modelo:\n")
    f.write(desvio_padrao_por_modelo_precision.to_string())


# Boxplot da Acurácia
# CORREÇÃO: Uso da API orientada a objetos do Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
df_resultados_acc.boxplot(ax=ax)

ax.set_xticklabels(df_resultados_acc.columns, rotation=45, ha='right')
plt.tight_layout()

nome_arquivo_acc = 'boxplot_acc.png'
caminho_completo_acc = os.path.join(caminho_acuracia, nome_arquivo_acc)
plt.savefig(caminho_completo_acc, dpi=300)
plt.close()

# Boxplot do Recall
# CORREÇÃO: Uso da API orientada a objetos do Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
df_resultados_recall.boxplot(ax=ax)

ax.set_xticklabels(df_resultados_recall.columns, rotation=45, ha='right')
plt.tight_layout()

nome_arquivo_recall = 'boxplot_recall.png'
caminho_completo_recall = os.path.join(caminho_recall, nome_arquivo_recall)
plt.savefig(caminho_completo_recall, dpi=300)
plt.close()

# Boxplot da Precisão
# CORREÇÃO: Uso da API orientada a objetos do Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
df_resultados_precision.boxplot(ax=ax)

ax.set_xticklabels(df_resultados_precision.columns, rotation=45, ha='right')
plt.tight_layout()

nome_arquivo_precision = 'boxplot_precision.png'
caminho_completo_precision = os.path.join(caminho_precisao, nome_arquivo_precision)
plt.savefig(caminho_completo_precision, dpi=300)
plt.close()

print("Spot-checking Finalizado")