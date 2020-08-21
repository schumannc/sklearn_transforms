from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class AggFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        notas_cols = ['NOTA_DE', 'NOTA_EM', 'NOTA_MF', 'NOTA_GO']
        rep_cols = ['REPROVACOES_DE', 'REPROVACOES_EM', 'REPROVACOES_MF', 'REPROVACOES_GO']
        
        notas = df[notas_cols].mean(axis=1)
        total_rep = df[rep_cols].sum(axis=1)
        
        notas = (notas / 10).clip(0, 1)
        reprovacoes = (total_rep / (3 * 4)).clip(0, 1)
        h_aula = (df["H_AULA_PRES"] / 25).clip(0, 1)
        faltas = (df["FALTAS"] / 8).clip(0, 1)
        tarefas = (df["TAREFAS_ONLINE"] / 7).clip(0, 1)
        
        df['SCORE_BOM'] = (notas + h_aula + tarefas) / 3
        df['SCORE_RUIM'] = (reprovacoes + faltas) / 2

        return df