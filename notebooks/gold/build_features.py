"""Notebook entry point for creating Gold aggregations."""
import pandas as pd
import os
import sys
from pathlib import Path
# Bibliotecas básicas
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame

def data_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes outliers from the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
            pd.DataFrame: Loaded dataset.
    """
    # ------------------------------------------------------------
    # Atividade 5 — Outliers
    # ------------------------------------------------------------

    print(f"\n{'='*80}")
    print("---------Tratamento de Outliers-------------")
    print(f"{'='*80}")

    variaveis_outlier = [
        "consumo_energia_kwh",
        "vibracao_motor_mm_s",
        "tempo_setup_min",
        "temperatura_c",
        "pressao_bar"
    ]

    # TODO 2: criar uma função para calcular limites por IQR
    def limites_iqr(serie, fator=1.5):

        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)

        IQR = Q3 - Q1

        limite_inferior = Q1 - fator * IQR
        limite_superior = Q3 + fator * IQR

        return limite_inferior, limite_superior
    # TODO 3: aplicar a função às variáveis e avaliar os limites
    def check_outliers(variaveis_outlier):
        resumo_outliers = []

        for col in variaveis_outlier:

            li, ls = limites_iqr(df[col])

            mask_outlier = (
                (df[col] < li) |
                (df[col] > ls)
            )

            qtd_outliers = mask_outlier.sum()

            pct_outliers = (
                qtd_outliers / len(df)
            ) * 100

            resumo_outliers.append({
                "Variável": col,
                "Limite Inferior": round(li, 2),
                "Limite Superior": round(ls, 2),
                "Qtd Outliers": qtd_outliers,
                "% Outliers": round(pct_outliers, 2)
            })

        resumo_outliers = pd.DataFrame(resumo_outliers)

        print("\nResumo de Outliers:")
        print(resumo_outliers)

    check_outliers(variaveis_outlier)

    # TODO 4: decidir a estratégia de tratamento
    print(f"\n{'='*80}")
    print("---------Justificativa-------------")
    print("Como os dados já foram tratados em termos de valores plausiveis e ausentes opto pela limitar por winsorização, uma vez que a regressão linear é sensível a outliers")
    print(f"{'='*80}")


    for col in variaveis_outlier:

        li, ls = limites_iqr(df[col])

        df[col] = df[col].clip(
            lower=li,
            upper=ls
        )
    check_outliers(variaveis_outlier)

    return df

def data_colinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes multicollinearity from the dataset.

    Args:
        df (pd.DataFrame): Input dataset.


    """
    # ------------------------------------------------------------
    # Atividade 6 — Variáveis redundantes e irrelevantes
    # ------------------------------------------------------------

    # TODO 3: decidir colunas a remover
    colunas_remover = [
        "registro_id",
        "codigo_campanha",
        "ruido_aleatorio",
        "energia_sensor_b_kwh"
    ]


    df = df.drop(
        columns=colunas_remover
    )

    return df

def data_symbolic_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts symbolic variables to numeric.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with symbolic variables converted to numeric.
    """


    # ------------------------------------------------------------
    # Atividade 7 — Transformação simbólico-numérica
    # ------------------------------------------------------------

    # TODO: definir a base final de modelagem

    # Sugestão de colunas categóricas após limpeza

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    # Variáveis categóricas
    variaveis_categoricas = [
        "linha_producao_pad",
        "turno_pad",
        "maquina"
    ]

    # Pipeline para categóricas
    transformador_categorico = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # ColumnTransformer
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "cat",
                transformador_categorico,
                variaveis_categoricas
            )
        ],
        remainder="passthrough"
    )

    # Aplicar transformação
    dados_transformados = preprocessador.fit_transform(df)

    # Recuperar nomes das colunas geradas pelo OneHotEncoder
    nomes_ohe = (
        preprocessador
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(variaveis_categoricas)
    )

    # Colunas que não foram transformadas
    outras_colunas = [
        col for col in df.columns
        if col not in variaveis_categoricas
    ]

    # Criar DataFrame final
    df_ohe = pd.DataFrame(
        dados_transformados.toarray() if hasattr(dados_transformados, "toarray") else dados_transformados,
        columns=list(nomes_ohe) + outras_colunas,
        index=df.index
    )


    df = df_ohe.copy()

    return df

def data_numeric_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts numeric variables to numeric format.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with numeric variables converted to numeric format.
    """

    # ------------------------------------------------------------
    # Atividade 8 — Transformação numérica-numérica
    # ------------------------------------------------------------

    print(f"\n{'='*80}")
    print("---------Transformação numérica-numérica-------------")
    print(f"{'='*80}")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    import numpy as np
    import pandas as pd

    # Sugestão de colunas numéricas
    colunas_numericas = [
        "idade_maquina_anos",
        "temperatura_c",
        "pressao_bar",
        "vibracao_motor_mm_s",
        "velocidade_esteira_m_min",
        "umidade_pct",
        "tamanho_lote",
        "tempo_setup_min",
        "paradas_nao_planejadas",
        "taxa_defeitos_pct"
    ]


    # Variável para transformação logarítmica
    coluna_log = ["paradas_nao_planejadas"]

    # Variáveis para StandardScaler
    colunas_standard = [
        "idade_maquina_anos",
        "temperatura_c",
        "pressao_bar",
        "vibracao_motor_mm_s",
        "velocidade_esteira_m_min",
        "umidade_pct",
        "tamanho_lote",
        "tempo_setup_min",
        "taxa_defeitos_pct"
    ]

    # Pipeline logarítmico
    transformador_log = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("log", FunctionTransformer(np.log1p, validate=False))
        ]
    )

    # Pipeline StandardScaler
    transformador_standard = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    # Pré-processador
    preprocessador = ColumnTransformer(
        transformers=[
            ("log", transformador_log, coluna_log),
            ("std", transformador_standard, colunas_standard)
        ],
        remainder="passthrough"
    )

    # Aplicar transformação
    dados_transformados = preprocessador.fit_transform(df)

    # Colunas não transformadas
    outras_colunas = [
        col for col in df.columns
        if col not in coluna_log + colunas_standard
    ]

    # Nome final das colunas
    nomes_colunas = (
        coluna_log +
        colunas_standard +
        outras_colunas
    )

    # DataFrame final
    df_transformado = pd.DataFrame(
        dados_transformados,
        columns=nomes_colunas,
        index=df.index
    )

    return df_transformado


def main(df=None, table_name=None, spark_session = None) -> any:
    """
    Main function to preprocess the dataset.
    """
    if df is None:
        df = spark.table("datapulse.silver."+table_name)

    if isinstance(df, SparkDataFrame):
        df = df.toPandas()

    df = data_outliers(df)
    df = data_colinearity(df)
    df = data_symbolic_to_numeric(df)
    df = data_numeric_to_numeric(df)

    if spark_session is None:
        df_spark = spark.createDataFrame(df)
    else:
        df_spark = spark_session.createDataFrame(df)

    df_spark.write.format("delta").mode("append").saveAsTable("datapulse.gold."+table_name)

    return df_spark


if __name__ == "__main__":
    main(df = None , table_name= "sensor_events")