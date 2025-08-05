import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

df = pd.read_csv('C:\\Users\\SERGIO\\Desktop\\EBAC\\clientes-v3-preparado.csv')

# Categorizar Salario: acima e abaixo da mediana
df['salario_categoria'] = (df['salario'] > df['salario'].median()).astype(int)  # 1 -Acima da mediana, 0 - abaixo ou igual a mediana

X = df[['idade', 'anos_experiencia', 'nivel_educacao_cod', 'area_atuacao_cod']]  # Preditar
Y = df['salario_categoria']  #Prever

# Dividir dados: Treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criar e treinar modelo - Regressão Logistica
modelo_lr = LogisticRegression()
modelo_lr.fit(X_train, Y_train)

# Criar e treinar modelo - Arvore de decisão
modelo_dt = DecisionTreeClassifier()
modelo_dt.fit(X_train, Y_train)

# Prever valores de teste
Y_prev_lr = modelo_lr.predict(X_test)
Y_prev_dt = modelo_dt.predict(X_test)

# Métricas de avaliação - Regressão Logística
accuracy_lr = accuracy_score(Y_test, Y_prev_lr)
precision_lr = precision_score(Y_test, Y_prev_lr)
recall_lr = recall_score(Y_test, Y_prev_lr)

print(f"\nAcurácia de regressão logística: {accuracy_lr:.2f}")
print(f"Precisão da regressão logística: {precision_lr:.2f}")
print(f"Recall (sensibilidade) da regressão logística: {recall_lr:.2f}")

# Métrica de Avaliação - Arvore de decisão
accuracy_dt = accuracy_score(Y_test, Y_prev_dt)
precision_dt = precision_score(Y_test, Y_prev_dt)
recall_dt = recall_score(Y_test, Y_prev_dt)

print(f"\nAcurácia de regressão logística: {accuracy_dt:.2f}")
print(f"Precisão da regressão logística: {precision_dt:.2f}")
print(f"Recall (sensibilidade) da regressão logística: {recall_dt:.2f}")

# Salvar modelo treinado
joblib.dump(modelo_lr, 'modelo_regressao_logistica.pkl')
joblib.dump(modelo_dt, 'modelo_arvore_decisao.pkl')
