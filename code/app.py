import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from sklearn.metrics import confusion_matrix

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'
target_col = 'shot_made_flag'

st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f'Acertos Kobe')

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)


st.title('Predições em Teste')
st.write(df_dev)

st.title('Predições em Produção')
st.write(df_prod)

fignum = plt.figure(figsize=(6,4))

st.title('Histograma de Probabilidade de Acerto (Teste x Produção)')
sns.histplot(df_dev.prediction_score_1, #corrigir nome da coluna, pode ser corrigido***
            label='Teste',
            ax=plt.gca())
sns.histplot(df_prod.predict_score,
            label='Produção',
            ax=plt.gca())

plt.title('Monitoramento Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de Acerto')
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)

st.title('Relatório de Classificação (Desenvolvimento)')
classification_report = metrics.classification_report(df_dev[target_col], df_dev.prediction_label)
classification_report_table = pd.DataFrame(metrics.classification_report(df_dev[target_col], df_dev.prediction_label, output_dict=True)).transpose()
classification_report_table.style.background_gradient(cmap='coolwarm')
st.write(classification_report_table)

st.title('Relatório de Classificação (Produção)')
classification_report = metrics.classification_report(df_prod[target_col], df_prod.predict_score > 0.5)
classification_report_table = pd.DataFrame(metrics.classification_report(df_prod[target_col], df_prod.predict_score > 0.5, output_dict=True)).transpose()
classification_report_table.style.background_gradient(cmap='coolwarm')
st.write(classification_report_table)

fignum = plt.figure(figsize=(6,4))
ypred_matrix = df_prod.predict_score > 0.5
yprod_matrix = df_prod.shot_made_flag > 0.5
cm = confusion_matrix(yprod_matrix, ypred_matrix)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz e Confusão - Árvore de Decisão (Produção)')
st.pyplot(fignum)

fignum = plt.figure(figsize=(6,4))
ypred_matrix = df_dev.prediction_label
yprod_matrix = df_dev.shot_made_flag
cm = confusion_matrix(yprod_matrix, ypred_matrix)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz e Confusão - Árvore de Decisão (Desenvolvimento)')
st.pyplot(fignum)




# para testar executar
# $ python app.py

# conda activate infnet-nn3
# $ streamlit run app.py
