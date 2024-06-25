import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import plotly_express as px
import numpy as np

st.set_page_config(layout="wide")

# Carregar os dados
df1 = pd.read_csv("clientes.csv", sep=";")
df2 = pd.read_csv("formas_pagamento.csv", sep=";")
df3 = pd.read_csv("produtos.csv", sep=";")
df4 = pd.read_csv("produtos_vendidos.csv", sep=";")
df5 = pd.read_csv("venda.csv", sep=";")

# Juntar os dados
df_venda_forma_pagamento = pd.merge(df5, df2, how='left', left_on='codigo_forma_pagamento', right_on='codigo_forma_pagamento')
df_venda_forma_cliente = pd.merge(df_venda_forma_pagamento, df1, how='left', left_on='cliente', right_on='Codigo')
df_venda_completa = pd.merge(df_venda_forma_cliente, df4, how='left', left_on='numero_venda', right_on='numero_venda')
df_final = pd.merge(df_venda_completa, df3, how='left', left_on='codigo_produto', right_on='codigo_produto')

# Limpar os dados
df_final['data_nascimento'] = pd.to_datetime(df_final['data_nascimento'], errors="coerce")
df_final['DATA'] = pd.to_datetime(df_final['DATA'], errors="coerce")
df_final['genereo'] = df_final['genereo'].astype('string')
df_final['cidade'] = df_final['cidade'].astype('string')
df_final['setor'] = df_final['setor'].astype('string')
df_final['forma_pagamento'] = df_final['forma_pagamento'].astype('string')
df_final['avista'] = df_final['avista'].astype('string')
df_final['produto'] = df_final['produto'].astype('string')
df_final['secao'] = df_final['secao'].astype('string')
df_final['ncm'] = pd.to_numeric(df_final['ncm'], errors="coerce")
df_final['idade'] = df_final['data_nascimento'].apply(lambda x: (datetime.datetime.now() - x).days // 365 if pd.notnull(x) else None)

df_final = df_final.dropna()

#calcular
df_final["Mes"] = df_final["DATA"].apply(lambda x: str(x.year) + "-" + str(x.month))

def filtrar_por_cidade_setor(df, cidade_selecionada, setor_selecionado):
    df_filtrado = df_final[df_final['cidade'] == cidade_selecionada]
    if setor_selecionado:
        df_filtrado = df_filtrado[df_filtrado['setor'] == setor_selecionado]
    return df_filtrado

def filtrar_por_ano_mes(df_final, ano_selecionado, mes_selecionado):
    df_filtrado = df_final[df_final['DATA'].dt.year == ano_selecionado]
    if mes_selecionado:
        df_filtrado = df_filtrado[df_filtrado['DATA'].dt.month == mes_selecionado]
    return df_filtrado

# Sidebar
with st.sidebar:
    tipo_filtro = st.selectbox("Tipo de Filtro", ["Cidade", "Ano"])
    if tipo_filtro == "Cidade":
        cidade_selecionada = st.selectbox("Cidade", df_final['cidade'].unique())
        setores_disponiveis = df_final[df_final['cidade'] == cidade_selecionada]['setor'].unique()
        setor_selecionado = st.selectbox("Setor", setores_disponiveis, key="setor")
    else:
        ano_selecionado = st.selectbox("Ano", df_final['DATA'].dt.year.unique(), key="ano")
        meses_disponiveis = df_final[df_final['DATA'].dt.year == ano_selecionado]['DATA'].dt.month.unique()
        mes_selecionado = st.selectbox("Mês", meses_disponiveis, key="mes")


# Aplicar filtro
if tipo_filtro == "Cidade":
    df_filtrado = filtrar_por_cidade_setor(df_final.copy(), cidade_selecionada, setor_selecionado)
else:
    df_filtrado = filtrar_por_ano_mes(df_final.copy(), ano_selecionado, mes_selecionado)

# Organizando colunas
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)
col9, col10 = st.columns(2)
col11, col12 = st.columns(2)

# Gráfico por idade e gênero
df_agrupado = df_filtrado.groupby(["genereo", "idade"]).size().to_frame(name="quantidade")
df_agrupado = df_agrupado.reset_index()
fig_date = px.bar(df_agrupado, x="idade", y="quantidade", color="genereo", title="Quantidade por idade e gênero")
col1.plotly_chart(fig_date, use_container_width=True)

# Agrupar os dados por mês e cidade
df_agrupado_cidade = df_filtrado.groupby(["Mes", "cidade"])["quantidade"].sum().to_frame(name="quantidade_vendida")
df_agrupado_cidade = df_agrupado_cidade.reset_index()
fig_cidade = px.bar(df_agrupado_cidade, x="cidade", y="quantidade_vendida", color="Mes", title="Quantidade de produtos vendidos por cidade e mês")
col2.plotly_chart(fig_cidade, use_container_width=True)

fig_kind = px.pie(df_filtrado, values="quantidade", names="produto",
                   title="Faturamento por tipo de pagamento")
col3.plotly_chart(fig_kind, use_container_width=True)

fig_hist = px.histogram(df_filtrado, x='forma_pagamento', title='Frequência de Formas de Pagamento Utilizadas')
col4.plotly_chart(fig_hist, use_container_width=True)

fig_scatter = px.scatter(df_filtrado, x='custo_atual', y='preco_atual', title='Relação entre Custo e Preço dos Produtos', hover_data=['produto', 'secao'])
col5.plotly_chart(fig_scatter, use_container_width=True)

fig_line = px.line(df_filtrado, x='secao', y='preco_atual', title='Variação do Preço Atual dos Produtos por Seção', markers=True, color='secao')
col6.plotly_chart(fig_line, use_container_width=True)


# Medidas de tendência central
media_idade = df_filtrado['idade'].mean()
mediana_idade = df_filtrado['idade'].median()
moda_idade = df_filtrado['idade'].mode()[0]

col7.metric("Média de Idade", f"{media_idade:.2f}")
col7.metric("Mediana de Idade", f"{mediana_idade:.2f}")
col7.metric("Moda de Idade", f"{moda_idade:.2f}")

media_valor_venda = df_filtrado['valor'].mean()
mediana_valor_venda = df_filtrado['valor'].median()
moda_valor_venda = df_filtrado['valor'].mode()[0]

col8.metric("Média de Valor de Venda", f"{media_valor_venda:.2f}")
col8.metric("Mediana de Valor de Venda", f"{mediana_valor_venda:.2f}")
col8.metric("Moda de Valor de Venda", f"{moda_valor_venda:.2f}")

# Modelo preditivo
X = df_filtrado[['idade', 'codigo_produto', 'quantidade', 'valor_unitario', 'desconto_aplicado']]
y = df_filtrado['quantidade']  #variável alvo: quantidade de produtos vendidos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #dividir os dados em treino e teste
model = LinearRegression() #treinar o modelo de regressão linear
model.fit(X_train, y_train)
y_pred = model.predict(X_test) #previsões
mse = mean_squared_error(y_test, y_pred)#fazer avaliação do modelo
r2 = r2_score(y_test, y_pred)

col9.write(f"Erro quadrático médio: {mse:.2f}")
col10.write(f"R2 Score: {r2:.2f}")

# Prever a quantidade de produtos vendidos para cada produto
df_filtrado['quantidade_predita'] = model.predict(df_filtrado[['idade', 'codigo_produto', 'quantidade', 'valor_unitario', 'desconto_aplicado']])

# Listar os top 10 produtos com maior quantidade prevista
top10_predicoes = df_filtrado.nlargest(10, 'quantidade_predita')[['codigo_produto', 'produto', 'quantidade_predita']]

#listagem
st.write("Top 10 produtos com maior quantidade prevista de vendas")
st.write(top10_predicoes)

#grafico
fig_top10 = px.bar(top10_predicoes, x='produto', y='quantidade_predita', title=f"Top 10 produtos com maior quantidade prevista de vendas")
st.plotly_chart(fig_top10)
