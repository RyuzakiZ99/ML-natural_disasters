# Trabalho de Aprendizado de Máquina - Etapa 1

O objetivo deste trabalho é a realização da caracterização de um problema de modelagem preditiva de classificação e a implementação e execução de *spot-checking* para a tarefa escolhida.

Para esta aplicação foi escolhido um dataset relacionado à previsão da possibilidade de ocorrência de tsunami considerando os dados de um terremoto. o dataset original está disponível [aqui](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/data).

## Como Rodar o Programa - Linux

Para apenas iniciar o ambiente virtual e iniciar:

```make```

Para realizar a inicialização adequada e rodar o programa:

```make run```

Para remover ambiente virtual e demais diretórios criados:

```make clean```

## Como Rodar o Programa - Windows

1. Criar o ambiente virtual:

```python3 -m venv venv```

2. Ativar o ambiente virtual:

```.\venv\Scripts\Activate.ps1```

3. Instalar as dependencias:

```pip install -r requirements.txt```

4. Agora pode-se rodar a análise EDA e o spot-checking separadamente.

Para rodar a análise EDA:

```python3 eda.py```

Para rodar o spot-checking:

```python3 spot_checking.py```

5. Por fim, para desativar o ambiente virtual criado:

```deactivate```