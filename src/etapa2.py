import os
import math
import optuna
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING) # Evita os logs no terminal gerados pelo optuna

# -------------------- Definições de Diretórios e Dataset --------------------

print("----------------------------------------")
print("Etapa 2")
print("----------------------------------------")

DIRETORIO_ETAPA2 = 'etapa_2'
DIRETORIO_OTIMIZACAO = 'otimizacao'
DIRETORIO_AVALIACAO = 'avaliacao'
DIRETORIO_INTERPRETACAO = 'interpretacao'

DIRETORIO_MATRIZ = 'matrizes_confusao'
DIRETORIO_PLOTS = 'plots_metricas'
DIRETORIO_PR = 'curvas_pr'

DATASET = 'earthquake_data_tsunami.csv'

# -------------------- Criação dos Diretórios --------------------

if not os.path.exists(DIRETORIO_ETAPA2):
    os.makedirs(DIRETORIO_ETAPA2)

caminho_oti = os.path.join(DIRETORIO_ETAPA2, DIRETORIO_OTIMIZACAO)
if not os.path.exists(caminho_oti):
    os.makedirs(caminho_oti)

caminho_ava = os.path.join(DIRETORIO_ETAPA2, DIRETORIO_AVALIACAO)
if not os.path.exists(caminho_ava):
    os.makedirs(caminho_ava)

caminho_int = os.path.join(DIRETORIO_ETAPA2, DIRETORIO_INTERPRETACAO)
if not os.path.exists(caminho_int):
    os.makedirs(caminho_int)

caminho_matriz = os.path.join(caminho_ava, DIRETORIO_MATRIZ)
if not os.path.exists(caminho_matriz):
    os.makedirs(caminho_matriz)

caminho_pr = os.path.join(caminho_ava, DIRETORIO_PR)
if not os.path.exists(caminho_pr):
    os.makedirs(caminho_pr)

# -------------------- Melhores Modelos da Etapa Anterior --------------------

modelos = ['KNN', 'Naive Bayes', 'Árvore de Decisão']

# -------------------- Otimização de Hiperparâmetros --------------------

print("Realizando a Otimização dos Hiperparâmetros")

# Carregamento dos Dados
df_original = pd.read_csv(DATASET)
remover_colunas = ['nst', 'dmin', 'gap', 'Year', 'Month']
df = df_original.drop(columns=remover_colunas, axis=1)

# Separação dos Atributos
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divisão dos Dados em Treinamento + Validação (CV) e Teste
x_cv, x_test, y_cv, y_test = train_test_split(x, y, test_size = 0.2, random_state = 61, stratify = y)

# Definição dos Folds
folds = 5
skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 42)

# Função do Optuna para Descobrir o Melhor Modelo
def objective (trial, classifier_name, x_cv, y_cv, skf):

    if classifier_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    elif classifier_name == 'Naive Bayes':
        var_smoothing = trial.suggest_float('var_smoothing', 1e-10, 1e-9, log=True)
        modelo = GaussianNB(var_smoothing=var_smoothing)

    else: # Árvore de Decisão
        max_depth = trial.suggest_int('max_depth', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
        modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', modelo)])

    scores = cross_val_score(estimator=pipeline, X=x_cv, y=y_cv, cv=skf, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()

melhores_resultados = {}

# Chama a Função de Orimização para Cada Modelo Individualmente
for nome in modelos:
    print(f"Otimizando: {nome}")

    study = optuna.create_study(direction="maximize")
    study.optimize((lambda trial: objective(trial, nome, x_cv, y_cv, skf)), n_trials=200)
    
    best_params = study.best_params

    melhores_resultados[nome] = {'params': best_params}

# Salva Valores das Otimizações em um Arquivo
caminho_optuna = os.path.join(caminho_oti, 'resultados_otimizacao.txt')
conteudo_oti = "Relatório da Otimização de Hiperparâmetros (Optuna)"

for nome, dados in melhores_resultados.items():
    conteudo_oti += f"Modelo: {nome}\n"
    conteudo_oti += "  Melhores Parâmetros:\n"
    for param, value in dados['params'].items():
        conteudo_oti += f"    - {param}: {value}\n"
    conteudo_oti += "\n"

with open(caminho_optuna, 'w') as f:
    f.write(conteudo_oti)

print("----------------------------------------")

# -------------------- Avaliação de Desempenho --------------------

print("Realizando a Análise de Desempenho")

# Definições para a análise das métricas
resultados_acuracia = {nome: [] for nome in modelos}
resultados_recall = {nome: [] for nome in modelos}
resultados_precisao = {nome: [] for nome in modelos}

caminho_res_ava = os.path.join(caminho_ava, 'resultados_avaliacao.txt')
conteudo_ava = "Relatório da Avaliação dos Modelos\n"

for nome, dados in melhores_resultados.items():
    if nome == 'KNN':
        modelo = KNeighborsClassifier(n_neighbors=dados['params']['n_neighbors'])

    elif nome == 'Naive Bayes':
        modelo = GaussianNB(var_smoothing=dados['params']['var_smoothing'])

    elif nome == 'Árvore de Decisão':
        modelo = DecisionTreeClassifier(max_depth=None, min_samples_split=dados['params']['min_samples_split'],random_state=42)

    pipeline_final = Pipeline([('scaler', MinMaxScaler()), ('model', modelo)])
    
    pipeline_final.fit(x_cv, y_cv)
    y_pred = pipeline_final.predict(x_test)

    # Cálculo das Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Geração da Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    label = ['Sem Tsunami (0)', 'Com Tsunami (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)

    plt.title(f'Matriz de Confusão - {nome}')
    plt.ylabel('Valores Reais')
    plt.xlabel('Valores Preditos')

    nome_plt_matriz = f'matriz_{nome.replace(" ", "-").lower()}.png'
    caminho_plt_matriz = os.path.join(caminho_matriz, nome_plt_matriz)
    plt.savefig(caminho_plt_matriz, dpi=300, bbox_inches='tight')
    plt.close()

    # Guardar Valores das Métricas
    conteudo_ava += f"Modelo: {nome}\n"
    conteudo_ava += f"- Acurácia: {acuracia}\n"
    conteudo_ava += f"- Precisão: {precisao}\n"
    conteudo_ava += f"- Recall: {recall}\n"

    resultados_acuracia[nome].append(acuracia)
    resultados_precisao[nome].append(precisao)
    resultados_recall[nome].append(recall)

    # Para os Gráficos da Curva PR
    y_probs = modelo.predict_proba(x_test.values)[:, 1]
    
    display = PrecisionRecallDisplay.from_predictions(y_test, y_probs)
    display.ax_.set_title(f"Curva Precision-Recall do {nome}")

    caminho_pr_modelo = os.path.join (caminho_pr, f'curva_pr_{nome}')
    plt.savefig(caminho_pr_modelo, dpi=300)

with open(caminho_res_ava, 'w') as f:
    f.write(conteudo_ava)

# Gráfico de Métricas
df_metricas = pd.DataFrame({
    'Modelo': modelos,
    'Acuracia': [resultados_acuracia[m][0] for m in modelos],
    'Precisao': [resultados_precisao[m][0] for m in modelos],
    'Recall': [resultados_recall[m][0] for m in modelos]
})

df_metricas_plot = df_metricas.melt(id_vars='Modelo', var_name='Metrica', value_name='Valor')

plt.figure(figsize=(12, 7))
sns.barplot(x='Modelo', y='Valor', hue='Metrica', data=df_metricas_plot, palette='Paired')

plt.title('Comparação de Desempenho dos Modelos', fontsize=16)
plt.ylabel('Valor da Métrica', fontsize=12)
plt.xlabel('Modelo', fontsize=12)
plt.legend(title='Métrica', loc='lower right')

caminho_plots_metricas = os.path.join(caminho_ava, 'plot_todas_metricas.png')
plt.savefig(caminho_plots_metricas, dpi=300)

print("----------------------------------------")

# -------------------- Interpretação do Modelo Final --------------------

print("Realizando a Interpretação do Modelo")

colunas_individuais = ['magnitude', 'cdi', 'mmi', 'sig', 'depth', 'latitude', 'longitude']
conteudo_inter = 'Resultados da Interpretação de Modelos - Previsão com Atributos Sendo Utilizados Sozinhos\n'

resultados_acuracia_colunas = {atributo: [] for atributo in colunas_individuais}
resultados_recall_colunas = {atributo: [] for atributo in colunas_individuais}
resultados_precisao_colunas = {atributo: [] for atributo in colunas_individuais}

# Testa Relação de Cada Atributo - O Quão Relevante ele é
for atributo in colunas_individuais:
    colunas_usadas = [atributo, 'tsunami']
    df_teste = df_original[colunas_usadas].copy()

    x = df_teste[[atributo]]
    y = df_teste['tsunami']

    x_train_inter, x_test_inter, y_train_inter, y_test_inter = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

    scaler = MinMaxScaler()
    scaler.fit(x_train_inter)
    x_train_inter = scaler.transform(x_train_inter)
    x_test_inter = scaler.transform(x_test_inter)

    modelo = DecisionTreeClassifier(max_depth=dados['params']['max_depth'], min_samples_split=dados['params']['min_samples_split'],random_state=42)

    modelo.fit(x_train_inter, y_train_inter)
    y_pred_inter = modelo.predict(x_test_inter)

    acuracia = accuracy_score(y_test_inter, y_pred_inter)
    precisao = precision_score(y_test_inter, y_pred_inter)
    recall = recall_score(y_test_inter, y_pred_inter)

    conteudo_inter += f"Atributo: {atributo}\n"
    conteudo_inter += f"     -- Acurácia: {acuracia}\n"
    conteudo_inter += f"     -- Precisão: {precisao}\n"
    conteudo_inter += f"     -- Recall: {recall}\n"

    resultados_acuracia_colunas[atributo].append(acuracia)
    resultados_precisao_colunas[atributo].append(precisao)
    resultados_recall_colunas[atributo].append(recall)

caminho_res_inter = os.path.join(caminho_int, 'resultados_interpretacao.txt')
with open(caminho_res_inter, 'w') as f:
    f.write(conteudo_inter)

# Gráfico das Métricas Individuais - para analisar relevância
df_inter = pd.DataFrame({
    'Atributo': colunas_individuais,
    'Acuracia': [resultados_acuracia_colunas[a][0] for a in colunas_individuais],
    'Precisao': [resultados_precisao_colunas[a][0] for a in colunas_individuais],
    'Recall': [resultados_recall_colunas[a][0] for a in colunas_individuais]
})

df_inter_plot = df_inter.melt(id_vars='Atributo', var_name='Metrica', value_name='Valor')

plt.figure(figsize=(12, 7))
sns.barplot(x='Atributo', y='Valor', hue='Metrica', data=df_inter_plot, palette='Paired')

plt.title('Comparação de Desempenho dos Atributos Individualmente', fontsize=16)
plt.ylabel('Valor da Métrica', fontsize=12)
plt.xlabel('Atributo', fontsize=12)
plt.legend(title='Métrica', loc='lower right')

caminho_plots_atributos = os.path.join(caminho_int, 'plot_atributos_individuais.png')
plt.savefig(caminho_plots_atributos, dpi=300)

print("----------------------------------------")

print("Processo Finalizado")