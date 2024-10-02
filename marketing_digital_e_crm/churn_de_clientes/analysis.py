import pandas as pd
import numpy as np
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


# Métricas de Desempenho
from scipy.stats import ks_2samp
from sklearn.metrics import (accuracy_score, 
                            f1_score, 
                            precision_score, 
                            recall_score, 
                            roc_auc_score)

from sklearn.model_selection import StratifiedKFold



class Analysis:
    def __init__(self, data):
        # Verifica se data é um DataFrame
        if not isinstance(data, pd.DataFrame):
          raise ValueError(f"O argumento 'data' deve ser um DataFrame do pandas, mas foi fornecido um objeto do tipo {type(data)}.")

        self.data = data.copy()
        self._results = {}

        numerical_columns = self.data.select_dtypes(include=['number']).columns

        
        for column in numerical_columns:
          if self.data[column].nunique() >= 10:
            self.set_grouping(column)
          else:
            self.data[column+'_bin'] = self.data[column].astype('str').astype('category')


    def set_order(self, column, order):
      self.data[column] = pd.Categorical(self.data[column], categories=order, ordered=True)



    def set_grouping(self, column, por = None, iniciar_em = None, finalizar_em = None):

      valor_min = min(self.data[column])
      valor_max = max(self.data[column])
     
      if iniciar_em is None:
        iniciar_em = valor_min

      if finalizar_em is None:
        finalizar_em = valor_max

      if por is None:
        por = (finalizar_em - iniciar_em) / 5

      iniciar_em = int(iniciar_em)
      finalizar_em = int(finalizar_em)
      por = int(por)


      # Definindo os limites das faixas
      bins = [i for i in range(iniciar_em, finalizar_em, por)]

      if iniciar_em > valor_min:
          bins = [valor_min] + bins

      if finalizar_em <= valor_max:
          bins = bins + [valor_max + 1]


      # Definindo os rótulos das faixas
      labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
      

      # Usando a função cut para categorizar os valores
      self.data[column+'_bin'] = pd.cut(self.data[column], bins=bins, labels=labels, right=False)


    def get_results(self):
      """
      Retorna os resultados de todas as análises.
      """
      return self._results

    def remove_column(self, column):
      """
      Remove uma variável da análise e seus resultados.

      Parâmetros:
      - column: Nome da coluna a ser removida.
      """
      if column in self._results:
        del self._results[column]
        self.data.drop(columns=[column], inplace=True)
        print(f"A coluna '{column}' foi removida")
      else:
        raise ValueError(f"A coluna '{column}' não foi encontrada nos resultados.")


    def get_table(self, column):
      """
      Retorna a tabela de análise para uma coluna específica.

      Parâmetros:
      - column: Nome da coluna para a qual obter a tabela de análise.
      """
      if column in self._results:
        return self._results[column]['table']
      else:
        raise ValueError(f"A coluna '{column}' não foi encontrada nos resultados.")




class BivariateAnalysis(Analysis):
    def __init__(self, data, binary_target):
      super().__init__(data)
      # Verifica se data é um DataFrame
      if not isinstance(data, pd.DataFrame):
        raise ValueError(f"O argumento 'data' deve ser um DataFrame do pandas, mas foi fornecido um objeto do tipo {type(data)}.")

      self.data = data.copy()
      self._results = {}
      self.binary_target = binary_target

      numerical_columns = self.data.select_dtypes(include=['number']).columns

      for column in numerical_columns:
        if column != self.binary_target:
          if self.data[column].nunique() >= 10:
            self.set_grouping(column)
          else:
            self.data[column+'_bin'] = self.data[column].astype('str').astype('category')


    def visualize_results(self, columns=None):
      """
      Visualiza as tabelas de análise para todas as variáveis.
      """
      filtered_results = {}
      if columns:
        # transforma a str em list
        columns = [columns] if type(columns) is str else columns
        filtered_columns = {key: self._results[key] for key in columns if key in self._results}
      else:
        filtered_columns = self._results

      for col, result in filtered_columns.items():

        # Transformar o DataFrame em uma tabela com tabulate
        table_str = tabulate(filtered_columns[col]['table'], headers='keys', tablefmt='rounded_grid')

        # Título Geral
        title = f"Resultados para {col} e {self.binary_target}"
        title = f"\n{title.center(len(table_str.splitlines()[0]))}\n"

        # Inserir o título após a tabulação
        print(title + table_str, "\n")


    def _calculate_bivariate(self, column):
      table = self.data.pivot_table(values=self.binary_target, index=column, aggfunc=['count', 'sum'], observed=False)

      table.columns = ['freq', 'sim']
      table['não'] = table['freq'] - table['sim']
      table['%freq'] = table['freq']/(table['freq'].sum())
      table['%sim'] = table['sim'] / table['sim'].sum()
      table['%não'] = table['não'] / table['não'].sum()
      table['%taxa_sim'] = table['sim'] / table['freq']

      return table
    


    def plot_bivariada(self, column, figsize=(12, 4)):
        
        if column+'_bin' in self.data.columns:
          column = column+'_bin'
        else:
          column

        # Criar o gráfico com eixos duplos
        fig, ax1 = plt.subplots(figsize=figsize)

        table = self._calculate_bivariate(column)

        mean_taxa_sim = table['sim'].sum() / table['freq'].sum()


        # Gráfico de barras horizontal no primeiro eixo x
        ax1.set_ylabel(table.index.name)
        ax1.set_xlabel('Frequência Absoluta', color='g')
        sns.barplot(y=table.index, x=table['freq'], color='g', ax=ax1)
        ax1.tick_params(axis='x', labelcolor='g')

        # Adicionar o segundo eixo x
        ax2 = ax1.twiny()
        ax2.set_xlabel(f'Proporção de {self.binary_target}', color='darkblue')
        sns.lineplot(x=table['%taxa_sim'], y=table.index, sort=False, marker='o', color='darkblue', linewidth=2, ax=ax2)
        ax2.tick_params(axis='x', labelcolor='darkblue')


        # Adicionar linha pontilhada para a média
        ax2.axvline(x=mean_taxa_sim, color='red', linestyle='--', label=f'Média: {mean_taxa_sim:.2%}')


        # Formatar o eixo x da taxa_sim para mostrar porcentagens
        ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax2.set_xlim(0, 1)  # Definir limites de 0% a 100%
        

        # Adicionar título
        plt.title(f'Proporção de {self.binary_target} por {table.index.name}')

        ax2.legend()
        
        # Ajustar layout para evitar sobreposição
        plt.tight_layout()

        # Mostrar o gráfico
        plt.show()


class InformationValue(BivariateAnalysis):
    def __init__(self, data, binary_target):
        super().__init__(data, binary_target)

    def calculate_iv(self):
      categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
      for col in categorical_columns:
        if col != self.binary_target:
          table, total = self._calculate_iv_column(col)
          self._results[col] = {'table': table, 'total': total}


    def _calculate_iv_column(self, column):

      table = self._calculate_bivariate(column)

      # Adicionando verificação para evitar divisão por zero
      table['Odds'] = np.where(table['%não'] == 0,(table['%sim'] / 0.01), table['%sim'] / table['%não'])
      table['WoE'] = table['Odds'].apply(lambda x: 0 if x == 0 else np.log(x))  # corrige problema do log(0) = -inf
      table['IV'] = (table['%sim'] - table['%não']) * table['WoE']

      # Excluindo linhas com frequencias zeradas criadas pelo set_grouping
      table.dropna(inplace=True)

      # Calculando os totais
      total = table[['freq', 'sim', 'não', '%freq', '%taxa_sim', 'IV']].sum()
      total['%freq'] = total['%freq']*100
      total['%taxa_sim'] = round(total['sim'] / total['freq']*100, 2)
      total['IV'] = round(total['IV'], 2)
      total.name = "Total"
      total = pd.DataFrame(total).T
      total.index.name = column


      #Arredonda para duas casas decimais
      for coluna in ['%sim',	'%não', '%freq', '%taxa_sim']:
        table[coluna] = table[coluna].map(lambda x: round(x*100, 2))

      for coluna in ['Odds',	'WoE',	'IV']:
        table[coluna] = table[coluna].map(lambda x: round(x, 2))


      return table, total


    def get_iv_total(self, column):
        """
        Retorna os totais da análise para uma coluna específica.

        Parâmetros:
        - column: Nome da coluna para a qual obter os totais da análise.
        """
        if column in self._results:
            return self._results[column]['total']
        else:
            raise ValueError(f"A coluna '{column}' não foi encontrada nos resultados.")

    def get_iv_predict(self):
      all_results = []

      for col, result in self._results.items():
          result_total = result['total'][['IV']].copy()
          result_total['Variável'] = col
          all_results.append(result_total)

      separacao = pd.concat(all_results).set_index('Variável')

      # Definir os intervalos e categorias
      intervalos = [float('-inf'), 0.02, 0.1, 0.3, 0.5, float('inf')]
      categorias = ['Muito fraco', 'Fraco', 'Médio', 'Forte', 'Muito Forte']

      # Aplicar pd.cut diretamente ao DataFrame
      separacao['poder_separação'] = pd.cut(separacao['IV'], bins=intervalos, labels=categorias, right=False)

      separacao.sort_values('IV', ascending=False, inplace=True)

      return separacao



    def visualize_results(self, columns=None, min_iv=None, max_iv=None):
      """
      Visualiza as tabelas de análise de Valor de Informação e os totais para todas as variáveis.
      """
      filtered_results = {}
      if columns:
        # transforma a str em list
        columns = [columns] if type(columns) is str else columns
        filtered_columns = {key: self._results[key] for key in columns if key in self._results}
      else:
        filtered_columns = self._results

      for col, result in filtered_columns.items():
        iv_total_value = result['total']['IV'].values
        if (min_iv is None or (iv_total_value >= min_iv)) and (max_iv is None or (iv_total_value <= max_iv)):
          # Juntar os DataFrames
          filtered_results[col] = pd.concat([result['table'], result['total']], axis=0, sort=False).fillna('')

          # Transformar o DataFrame em uma tabela com tabulate
          table_str = tabulate(filtered_results[col], headers='keys', tablefmt='rounded_grid')

          # Título Geral
          title = f"Resultados de Information Value para '{col}' e '{self.binary_target}'"
          title = f"\n{title.center(len(table_str.splitlines()[0]))}\n"

          # Inserir o título após a tabulação
          print(title + table_str, "\n")



    def save_iv_txt(self):
      for col, result in self._results.items():
        # Juntar os DataFrames
        results = pd.concat([result['table'], result['total']], axis=0, sort=False).fillna('')
        # Transformar o DataFrame em uma tabela com tabulate

        table_str = tabulate(results, headers='keys', tablefmt='rounded_grid')

        # Título Geral
        title = f"Resultados de Information Value para {col} e {self.binary_target}"
        title = f"\n{title.center(len(table_str.splitlines()[0]))}\n"

        # Concatenar as tabelas
        combined_table_str = title + table_str + '\n\n\n\n'


        # Salvar o resultado tabulado em um arquivo de texto
        with open('output.txt', 'a') as f:
            f.write(combined_table_str)


class ConfidenceInterval(BivariateAnalysis):
    def __init__(self, data, binary_target):
      super().__init__(data, binary_target)

    def calculate_prop_ci(self, confidence=0.95, total=False):
      categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
      for col in categorical_columns:
        if col != self.binary_target:
          table = self._calculate_prop_conf_interval(col, confidence=confidence, total=total)
          self._results[col] = {'table': table}

    def _calculate_prop_conf_interval(self, column, confidence, total):
      """
        Calcula o intervalo de confiança para a proporção de uma variável alvo agrupada por uma variável categórica em um DataFrame.
      """
      from scipy import stats

      table = pd.pivot_table(self.data, values=self.binary_target, index=column, aggfunc=['count', 'mean'], margins=total, margins_name='Total')

      table.columns = ['n_amostra', 'prop(%)']
      table['var'] = table['prop(%)'] * (1 - table['prop(%)'])

      t_stat = abs(stats.t.ppf((1 - confidence) / 2, table['n_amostra'] - 1)) # T-score para o intervalo de confiança

      #Para obter uma estimativa do erro padrão, basta dividir o desvio padrão pela raiz quadrada do tamanho amostral
      error = np.sqrt(table['var'] / table['n_amostra'] )

      # Limites do intervalo de confiança
      table['lim_inf(%)'] = table['prop(%)'] - (t_stat * error)
      table['lim_sup(%)'] = table['prop(%)'] + (t_stat * error)


      for col in table.columns[1:]:
        table[col] = round(table[col] *100 , 2)

      return table




# Definindo a função de tabelas de frequência

def frequency_table(df, variavel_qualitativa, metadados: dict = None, ordem = None, bins = None):
    """
    Gera tabelas de frequência absoluta, relativa e acumulada para uma variável qualitativa em um DataFrame.

    Parâmetros:
    - df: DataFrame pandas contendo os dados.
    - variavel_qualitativa: Nome da coluna que representa a variável qualitativa.
    - metadados: Dicionário opcional de metadados para mapear os valores da variável qualitativa.
    - ordem: Lista opcional que especifica a ordem desejada para o índice da tabela de frequência.
            Se None (padrão), assume-se que a variável é nominal e não existe ordenação dentre as categorias.
            Se uma lista é fornecida, a variável é tratada como ordinal e existe uma ordenação entre as categorias.
    - bins: Tamanho de faixas de agrupamento para variaveis quantitativas.

    Retorna:
    - DataFrame contendo tabelas de frequência com as colunas 'Frequência Absoluta', 'Frequência Relativa' e 'Frequência Acumulada'.
    """

    # Verifica se df é um DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"O argumento 'df' deve ser um DataFrame do pandas, mas foi fornecido um objeto do tipo {type(df)}.")

    # Verifica se a variável qualitativa está presente no DataFrame
    if variavel_qualitativa not in df.columns:
        raise ValueError(f"A coluna '{variavel_qualitativa}' não está presente no DataFrame.")

    dataframe_variavel_qualitativa = df[variavel_qualitativa]

    if metadados:
      dataframe_variavel_qualitativa = dataframe_variavel_qualitativa.map(metadados)

    if bins:
      sort=False
    else:
      sort=True

    # Calcula a frequência absoluta
    frequencia_absoluta = dataframe_variavel_qualitativa.value_counts(bins = bins, sort=sort)

    if isinstance(df[variavel_qualitativa].dtype, pd.CategoricalDtype):
      ordem = (list(df[variavel_qualitativa].dtype.categories.values))


    # Reindexa a frequencia_absoluta antes de calcular as outras frequências se uma ordem específica foi fornecida para o índice
    if ordem:
      # Verifica se os valores de ordem estão presente no index
      if not all(valor in ordem for valor in dataframe_variavel_qualitativa.unique()):
        raise ValueError(f"Pelo menos um valor de '{ordem}' está diferente ou não está em '{list(dataframe_variavel_qualitativa.unique())}'.")
      frequencia_absoluta = frequencia_absoluta.reindex(ordem)

    # Calcula as frequências relativa e acumulada
    frequencia_relativa = frequencia_absoluta / len(dataframe_variavel_qualitativa)
    frequencia_acumulada = frequencia_relativa.cumsum()

    # Formata as colunas 'Frequência Relativa' e 'Frequência Acumulada'
    formato_percentual = lambda x: f'{x:.2%}'

    tabela_frequencia = pd.DataFrame({
        'Frequência Absoluta': frequencia_absoluta,
        'Frequência Relativa': frequencia_relativa.map(formato_percentual),
        'Frequência Acumulada': frequencia_acumulada.map(formato_percentual)
    }).rename_axis(variavel_qualitativa)


    return tabela_frequencia


# Função r2
def r2(df, categorical, numerical):
    """
    Calcula o Coeficiente de Determinação (R²) para uma variável qualitativa e uma variável quantitativa em um DataFrame.

    Parâmetros:
    - df: DataFrame pandas contendo os dados.
    - categorical: Nome da coluna que representa a variável qualitativa.
      - Para calculo de R² combinado, informar uma lista de variáveis.
    - numerical: Nome da coluna que representa a variável quantitativa.

    Retorna:
    - Coeficiente de Determinação (R²) e nomes da categorical e numerical em uma tupla.
    """
    varp = lambda x: np.var(x, ddof=0)

    # Criar tabela de contagem e variância
    tabela = pd.pivot_table(df, values=numerical, index=categorical, aggfunc=['count', varp], margins=True, observed=True, margins_name='Total') #margins=True retorna os totais
    tabela.columns = ['N', 'varp']

    # Separando o total dos demais dados
    total = tabela.loc[[tabela.last_valid_index()]]

    # Filtrando os demais dados sem o total e removendo linhas NaN.
    tabela = tabela.drop(tabela.last_valid_index()).dropna()

    # Calcular variância ponderada total
    var_ponderada = (tabela['N'] @ tabela['varp']) / total['N'].values[0]

    # Calcular R²
    r2_valor = 1 - (var_ponderada / total['varp'].values[0])

    return categorical, numerical, r2_valor


def skf_split(X, y, scaler=None, **kwargs):
    skf = StratifiedKFold() # shuffle=True embaralha as amostras de cada classe antes de dividir em lotes. 
    indexes = skf.split(X, y)

    splits = []

    # Separando a base
    for train_index, test_index in indexes:
        X_train = X.iloc[train_index].values
        X_test = X.iloc[test_index].values

        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values


        if scaler:
            # Padronização da Escala usando conjunto de treino
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        splits.append({
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test
            })

    return splits


def skf_metrics(model, X, y, **kwargs):
    splits = skf_split(X=X, y=y)

    metrics = ['Target', 'Acurácia', 'AUROC', 'KS', 'Precision', 'Recall', 'F1']

    # Criando um dicionário onde cada chave é o nome da métrica e o valor é uma lista vazia
    train_metrics = {f"{metric}": [] for metric in metrics}
    test_metrics = {f"{metric}": [] for metric in metrics}

    for split in splits:
        # Separando a base    
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        
        
        #Treina o modelo
        model.fit(X_train, y_train)

        # Cálculo dos valores preditos
        ypred_train = model.predict(X_train)
        ypred_proba_train = model.predict_proba(X_train)[:,1]

        ypred_test = model.predict(X_test)
        ypred_proba_test = model.predict_proba(X_test)[:,1]

        # Target em Treino e Teste
        train_metrics['Target'].append(y_train.mean())
        test_metrics['Target'].append(y_test.mean())

        # Métricas de Desempenho
        train_metrics['Acurácia'].append(accuracy_score(y_train, ypred_train))
        test_metrics['Acurácia'].append(accuracy_score(y_test, ypred_test))
        
        train_metrics['AUROC'].append(roc_auc_score(y_train, ypred_proba_train))
        test_metrics['AUROC'].append(roc_auc_score(y_test, ypred_proba_test))
        
        train_metrics['KS'].append(ks_2samp(ypred_proba_train[y_train==1], ypred_proba_train[y_train!=1]).statistic)
        test_metrics['KS'].append(ks_2samp(ypred_proba_test[y_test==1], ypred_proba_test[y_test!=1]).statistic)
        
        train_metrics['Precision'].append(precision_score(y_train, ypred_train, zero_division=0))
        test_metrics['Precision'].append(precision_score(y_test, ypred_test, zero_division=0))
        
        train_metrics['Recall'].append(recall_score(y_train, ypred_train))
        test_metrics['Recall'].append(recall_score(y_test, ypred_test))
        
        train_metrics['F1'].append(f1_score(y_train, ypred_train))
        test_metrics['F1'].append(f1_score(y_test, ypred_test))
    
    # Criando DataFrames para treino e teste
    df_treino = pd.DataFrame(train_metrics)
    df_teste = pd.DataFrame(test_metrics)

    # Adicionando uma coluna para identificar se os dados são de treino ou teste
    df_treino['Base'] = 'Treino'
    df_teste['Base'] = 'Teste'

    df_combined = pd.concat([df_treino, df_teste], ignore_index=True)

    # Calculando a média e o desvio padrão agrupados por 'tipo'
    result = df_combined.groupby('Base').agg(['mean', 'std']).T
    result['Variação'] = abs((result['Teste'] - result['Treino']) / result['Treino'])
    return result[['Treino', 'Teste', 'Variação']]


# Função para cálculo do KS
def ks_stat(y, y_pred):
    return ks_2samp(y_pred[y==1], y_pred[y!=1]).statistic

# Função para cálculo do desempenho de modelos
def calculate_performance(modelo, x_train, y_train, x_test, y_test):
    
    # Cálculo dos valores preditos
    ypred_train = modelo.predict(x_train)
    ypred_proba_train = modelo.predict_proba(x_train)[:,1]

    ypred_test = modelo.predict(x_test)
    ypred_proba_test = modelo.predict_proba(x_test)[:,1]

    # Métricas de Desempenho
    acc_train = accuracy_score(y_train, ypred_train)
    acc_test = accuracy_score(y_test, ypred_test)
    
    roc_train = roc_auc_score(y_train, ypred_proba_train)
    roc_test = roc_auc_score(y_test, ypred_proba_test)
    
    ks_train = ks_stat(y_train, ypred_proba_train)
    ks_test = ks_stat(y_test, ypred_proba_test)
    
    prec_train = precision_score(y_train, ypred_train, zero_division=0)
    prec_test = precision_score(y_test, ypred_test, zero_division=0)
    
    recl_train = recall_score(y_train, ypred_train)
    recl_test = recall_score(y_test, ypred_test)
    
    f1_train = f1_score(y_train, ypred_train)
    f1_test = f1_score(y_test, ypred_test)

    df_desemp = pd.DataFrame({'Treino':[acc_train, roc_train, ks_train, 
                                        prec_train, recl_train, f1_train],
                              'Teste':[acc_test, roc_test, ks_test,
                                       prec_test, recl_test, f1_test]},
                            index=['Acurácia','AUROC','KS',
                                   'Precision','Recall','F1'])
    
    df_desemp['Variação'] = abs((df_desemp['Teste'] - df_desemp['Treino']) / df_desemp['Treino'])
    df_desemp.index.name = 'Métrica'
    
    return df_desemp

def cv_roc_auc_score(model, splits):
    aucs = {'Treino': [] , 'Teste': []}

    for split in splits:
        #Treina o modelo
        model.fit(split['X_train'], split['y_train'])

        pred_prob_train = model.predict_proba(split['X_train'])[:, 1]
        pred_prob_test = model.predict_proba(split['X_test'])[:, 1]

        roc_train = roc_auc_score(split['y_train'], pred_prob_train)
        roc_test = roc_auc_score(split['y_test'], pred_prob_test)

        aucs['Treino'].append(roc_train)
        aucs['Teste'].append(roc_test)
        
    aucs = pd.DataFrame(aucs)
    aucs['Variação'] = abs((aucs['Teste'] - aucs['Treino']) / aucs['Treino'])

    return aucs

def cv_recall_score(model, splits):
    recalls = {'Treino': [] , 'Teste': []}

    for split in splits:
        #Treina o modelo
        model.fit(split['X_train'], split['y_train'])

        ypred_train = model.predict(split['X_train'])
        ypred_test = model.predict(split['X_test'])


        recall_train = recall_score(split['y_train'], ypred_train)
        recall_test = recall_score(split['y_test'], ypred_test)

        recalls['Treino'].append(recall_train)
        recalls['Teste'].append(recall_test)
        
    recalls = pd.DataFrame(recalls)
    recalls['Variação'] = abs((recalls['Teste'] - recalls['Treino']) / recalls['Treino'])

    return recalls