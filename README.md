# Rossmann Store Sales Prediction

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/rossmann.jpeg" alt="image" width="2000">

# Rossmann Sales Model

O objetivo desse projeto é fornecer para o CFO da Rossmann Drug Stores, um **modelo de previsão de vendas** para as próximas seis semanas para que ele possa definir um orçamento específico para reformas nas lojas. O modelo de previsão atualmente utilizado não atende as necessidades da empresa, portanto, o modelo de machine learning desenvolvido nesse projeto veio como uma solução exata para esse problema de negócio.

O projeto foi desenvolvido através da técnica CRISP-DM, e ao final do primeiro ciclo de desenvolvimento foi possível produzir um modelo de previsão com indíce **MAPE Error de 11%** utilizando o algoritmo **XGBoost**.

Em termos de negócio, o resultado desse modelo de previsão pode ser resumido com os números abaixo:


| __Scenarios__ | __Values__ |
| ------------- | -----------|
| predictions	| €287,260,406.62 |
| worst scenario | €286,409,667.62 |
| best scenario	| €288,111,145.61 |

## 1. Sobre a Rossmann Drug Store

### 1.1 Contexto do negócio:

A Rossmann é uma das maiores redes de drogaria da Europa, com cerca de 56.200 funcionários e mais de 4000 lojas em diversos países, como Alemanha, Polônia, Hungria, República Tcheca, Turquia, Albânia, Kosovo e Espanha. É uma empresa com um grande sortimento de produtos que são oferecidos as seus clientes, incluindo produtos próprios. A companhia está em grande expansão e num ritmo elevado, com grandes investimentos.

### 1.2 Questão do negócio:

Em uma reunião com os líderes de cada departamento, o CFO da Rossmann fez uma proposta para reformar toda a loja.
Qual é a causa raiz do problema?

O CFO da Rossmann quer prever as próximas 6 semanas de vendas de cada loja, para poder antecipar parte dessa receita para renová-las.

### Quem é o Stakeholder do problema?

O CFO, que solicitou diretamente a resposta para o problema.

### Como será a solução?

Granularidade: vendas diárias durante 6 semanas/loja

Tipo de problema: previsão de vendas, problema de regressão

### Como a solução será entregue? 
API via Heroku

### 1.3 Sobre os dados:

Os dados foram disponibilizados pela empresa na plataforma do Kaggle: https://www.kaggle.com/c/rossmann-store-sales/data

|***Atributo*** | ***Descrição*** |
| -------- | --------- |
|**Id** | um Id que representa um (Store, Date) concatenado dentro do conjunto de teste |
|**Store** | um id único para cada loja |
|**Sales** | o volume de vendas em um determinado dia |
|**Customers** | o número de clientes em um determinado dia |
|**Open** | um indicador para saber se a loja estava aberta: 0 = fechada, 1 = aberta |
|**StateHoliday** | indica um feriado estadual. Normalmente todas as lojas, com poucas exceções, fecham nos feriados estaduais. Observe que todas as escolas fecham nos feriados e finais de semana. a = feriado, b = feriado da Páscoa, c = Natal, 0 = Nenhum |
|**SchoolHoliday** | indica se (Store, Date) foi afetada pelo fechamento de escolas públicas |
|**StoreType** | diferencia entre 4 modelos de loja diferentes: a, b, c, d | 
|**Assortment** | descreve um nível de sortimento: a = básico, b = extra, c = estendido | 
|**CompetitionDistance** | distância em metros até a loja concorrente mais próxima | 
|**CompetitionOpenSince[Month/Year]** | apresenta o ano e mês aproximados em que o concorrente mais próximo foi aberto| 
|**Promo** | indica se uma loja está fazendo uma promoção naquele dia | 
|**Promo2** | Promo2 é uma promoção contínua e consecutiva para algumas lojas: 0 = a loja não está participando, 1 = a loja está participando | 
|**Promo2Since[Year/Week]** | descreve o ano e a semana em que a loja começou a participar da Promo2 | 
|**PromoInterval** | descreve os intervalos consecutivos de início da promoção 2, nomeando os meses em que a promoção é iniciada novamente. Por exemplo. "Fev, maio, agosto, novembro" significa que cada rodada começa em fevereiro, maio, agosto, novembro de qualquer ano para aquela loja | 

### 1.4 Premissas do negócio:

- Os dias em que as lojas foram fechadas (Abertas) foram retirados da análise.
- Foram consideradas apenas lojas com valores de venda maiores que 0.
- Para as lojas que não possuíam informação de "Competition Distance", considerou-se que a distância deveria ser a maior distância observada no conjunto de dados.

## 2. Planejamento da solução:

O projeto foi desenvolvido através do método CRISP-DM, aplicando os seguintes passos:

**Passo 01 - Descrição dos dados:** Nessa etapa, o objetivo foi conhecer os dados, seus tipos, usar métricas estatísticas para identificar outliers no escopo do negócio e também analisar métricas estatísticas básicas como: média, mediana, máximo, mínimo, range, skew, kurtosis e desvio padrão. Nessa etapa também foram feitos alguns ajustes em features do dataset, como preenchimento de NA's por exemplo.

**Passo 02 - Feature Engineering:** Nessa etapa, foi desenvolvido um mapa mental para analisar o fenômeno, suas variáveis e os principais aspectos que impactam cada variável. A partir das características do hipóteses e da necessidade de novos atributos, foram elevados novos recursos a partir das variáveis originais, a fim de melhorar o fenômeno do ser modelado.

**Passo 03 - Filtragem dos dados:** O objetivo desta etapa foi filtrar linhas e excluir colunas que não são relevantes para o modelo ou não fazem parte do escopo do negócio, como por exemplo, desconsiderar dias que as lojas não estavam operando e/ou que não houveram vendas.

**Passo 04 - Análise Exploratória dos dados:** O objetivo desta etapa foi explorar os dados para encontrar insights, entender melhor a relevância das variáveis no aprendizado do modelo. Foram feitas analises univariadas, biváriadas e multivariadas, utilizandos os dados numéricos e categóricos do conjunto.

**Passo 05 - Preparação dos dados:** Nessa etapa,  os dados foram preparados para o inicio das aplicações de modelos de machine learning. Foram utilizadas técnicas como Rescaling e Transformation, através de encodings e nature transformation.

**Passo 06 - Seleção de Features:** O objetivo desta etapa foi selecionar os melhores atributos para treinar o modelo. Foi utilizado o algoritmo Boruta para fazer a seleção das variáveis, destacando as que tinham mais relevância para o fenômeno.

**Passo 07 - Modelagem de Machine Learning:** Nessa etapa foram feitos os testes e treinamento de alguns modelos de machine learning, onde foi possível comparar suas respectivas performance e feita a escolha do modelo ideal para o projeto. Inclusive foi utilizada a técnica de Cross Validation para garantir a performance real sobre os dados selecionados.

**Passo 08 - Hyperparameter Fine Tunning:** Tendo a escolha do algorotimo XBoost na etapa anterior, foi feita uma analise através do método Randon Search para escolher os melhores valores para cada um dos parâmetros do modelo. Ao final dessa etapa foi possível obter os valores finais da performance do modelo.

**Passo 09 - Tradução e interpretação de erros:** O objetivo dessa etapa foi de fato demonstrar o resultado do projeto, onde foi possível avaliar a performance do modelo com o viés de negócio, demonstrando o resultado financeiro que pode ser esperado se aplicado o modelo desenvolvido.

**Passo 10 - Deploy do modelo em produção:** Após execução bem sucedida do modelo, o objetivo foi publica-lo em um ambiente de nuvem para que outras pessoas ou serviços possam usar os resultados para melhorar a decisão de negócios. A plataforma de aplicativo em nuvem escolhida foi o Heroku.

**Passo 11 - Bot do Telegram:** A etapa final do projeto foi criar um bot no app de mensagens - Telegram, que possibilita consultar as previsões a qualquer momento e lugar, visto que também foi feito o deploy na plataforma em nuvem.

## 3. Principais insights:

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/H2.png" alt="image" width="2000">

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/H09.png" alt="image" width="2000">

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/H10.png" alt="image" width="2000">

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/H11.png" alt="image" width="2000">

<img src="https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/H12.png" alt="image" width="2000">


## 4. Performance dos Modelos de Machine Learning:

O dados do projeto foram testados com modelos lineares e não lineares.Foi utilizada a estratégia de selecionar 5 tipos de modelos: Modelo de média, dois modelos lineares, e dois não-lineares. A média por exemplo serviu como base de referência. Os modelos lineares servem para avaliar a complexidade de aprendizado do conjunto de dados. Caso a performance fosse ruim, poderia entender que seria necessário um modelo mais complexo. 

**- Modelos Lineares:**

   - Média
   - Linear Regression 
   - Linear Regression Regularized

**- Modelos Não Lineares:**

   - Random Forest Regressor 
   - XGBoost Regressor

**Comparação da performance dos modelos:**

***Model Name*** |	***MAE CV*** |	***MAPE CV*** |	***RMSE CV*** |
| ---------------- | ---------- | --------- | ---------- |
|Linear Regression	| 2081.73 +/- 295.63 |	0.3 +/- 0.02	| 2952.52 +/- 468.37
|Lasso |	2116.38 +/- 341.5 |	0.29 +/- 0.01	| 3057.75 +/- 504.26
|Random Forest Regressor	| 842.57 +/- 218.77 |	0.12 +/- 0.02	| 1264.08 +/- 321.73
|XGBoost Regressor |	1064.95 +/- 178.65	| 0.15 +/- 0.02	| 1519.92 +/- 242.12

**Performance final do modelo escolhido após Hyperparameter Fine Tuning:**

***Model Name*** | ***MAE*** | ***MAPE*** | ***RMSE*** |
| -------- | --------- | --------- | --------- |
|XGBoost Regressor | 760.056875 | 0.114527	 | 1088.444636 |


## 5. Resultado final - Model performance vs Business Values

O resultado final do projeto foi satisfatório para a maior parte das lojas abrangidas nos dados, conforme gráfico abaixo (Essas lojas em específico podem conter particularidades e possivelmente num segundo ciclo desse projeto, algo poderia ser feito para melhor a performance e predição para elas).

![image](https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/MAPE.png)

A maior parte das lojas tiveram o erro MAPE muito próximo do erro performado no modelo - **MAPE Error de 9%**

Como indicado no resumo prévio do projeto, o resultado que pode ser obtido utilizando-se do modelo, considerando o melhor e pior cenário, é o seguinte:

| __Scenarios__ | __Values__ |
| ------------- | -----------|
| predictions	| €287,260,406.62 |
| worst scenario | €286,409,667.62 |
| best scenario	| €288,111,145.61 |



Podemos observar o performance do modelo, avaliando a relação entre as vendas (dados de teste) e as predições:

![image](https://github.com/leomacedo86/Rossmann_Store_Sales_Prediction/blob/main/Images/Predict.png)

## 6. Conclusão

O projeto desenvolvido foi concluído com êxito, onde foi possível projetar as vendas das próximas semanas para que o CFO tenha informações reais para criar o budget das lojas, podendo consultar em tempo real cada predição.

Como sugestão para melhoria, testar o modelo usando a random forest, para compararmos a diferença e performance entre os modelos.




