# 💳 Credit Scoring – Modelo de Risco com Streamlit

Este projeto implementa um **modelo de Credit Scoring** utilizando **Machine Learning** e disponibiliza uma **interface interativa no Streamlit** para escoragem de novos clientes.

A solução **prediz a probabilidade de inadimplência** de clientes solicitantes de cartão de crédito, usando dados estruturados e um modelo treinado off-line.

---

## 🚀 Tecnologias Utilizadas

| Componente | Biblioteca / Ferramenta |
|------------|-------------------------|
| **Linguagem** | Python 3.9+ |
| **Modelo de Scoring** | Scikit-Learn (Logistic Regression Pipeline) |
| **Aplicação Web** | Streamlit |
| **Manipulação de Dados** | Pandas / NumPy |
| **Serialização do Modelo** | Pickle |

---

## 📁 Estrutura do Repositório

```text
streamlit-final/
├── model/
│   └── model_final.pkl      # Modelo final treinado (pipeline + regressão logística)
├── data/
│   └── credit_scoring_template.csv  # Template para upload no Streamlit
├── streamlit_app/
│   └── app.py               # Aplicação principal Streamlit
├── score_functions.py       # Funções auxiliares (load, preprocess, score)
├── notebooks/               # Notebooks utilizados no desenvolvimento
└── README.md                # Documentação do projeto

⚠️ Nota importante:
A base completa (credit_scoring.ftr) não está neste repositório, pois o arquivo é grande e não é necessário para rodar o app.

🧠 Modelo de Machine Learning

O modelo utiliza:

Pipeline com:

Tratamento de zeros estruturais

Imputação de valores faltantes

One-Hot Encoding para variáveis categóricas

Normalização numérica

Logistic Regression

Treinado sobre uma base de crédito com 15 safras e desempenho observado em 12 meses.

🎯 Target

mau — indica se o cliente foi ou não inadimplente.

📊 Métricas Avaliadas
Métrica	Base Treino	Base OOT
AUC ROC	~0.76	~0.72
KS	Avaliado	Avaliado
Gini	Avaliado	Avaliado
🎬 Demonstração em Vídeo

📹 Clique para assistir ao funcionamento do aplicativo:
(adicione o link do vídeo aqui depois de subir para o Drive/YouTube)

🎥 Exemplo a adicionar:
https://drive.google.com/SEU_LINK_AQUI

▶️ Como Executar o Projeto Localmente
1️⃣ Instale as dependências
pip install -r requirements.txt

2️⃣ Execute o Streamlit
streamlit run streamlit_app/app.py

3️⃣ Faça Upload do CSV e veja o score!

Use o arquivo de template disponível em:

data/credit_scoring_template.csv

📌 Uso do App

Faça upload de um arquivo CSV seguindo o template fornecido.

O sistema processa e aplica o modelo automaticamente.

O resultado exibe:

✔ probabilidade de inadimplência

✔ classificação prevista

✔ opção de download dos resultados

👨‍🏫 Projeto Acadêmico

Este projeto foi desenvolvido para a conclusão do módulo de Machine Learning aplicado a crédito, no curso da EBAC – Escola Britânica de Artes Criativas e Tecnologia.

🙋‍♂️ Autor

👤 Bruno Pimenta
🌐 GitHub: https://github.com/brunopimentafc

