"""Notebook entry point for cleaning Silver data."""
import pandas as pd
import os
# Bibliotecas básicas
import numpy as np
from datetime import datetime
from pyspark.sql import DataFrame as SparkDataFrame



def corrigir_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige os prefixos das colunas conforme o padrão de nomenclatura."""

    renomear = {
        "str_vibracao_motor_mm_s": "dec_vibracao_motor_mm_s",
        "big_tamanho_lote": "dec_tamanho_lote",
        "str_consumo_energia_kwh": "dec_consumo_energia_kwh",
    }

    return df.rename(
        columns={
            origem: destino
            for origem, destino in renomear.items()
            if origem in df.columns
        }
    )

def padronizar_data(x):

    if pd.isna(x):
        return None

    x = str(x).strip()

    formatos = [
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%m-%d-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S"
    ]

    for fmt in formatos:
        try:
            dt = datetime.strptime(x, fmt)
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        except ValueError:
            pass

    return None


def typing_data(df ,file_path: str) -> pd.DataFrame:
    """
    Corrects data types in the dataset.
    """

    df["str_vibracao_motor_mm_s"] = pd.to_numeric(
    df["str_vibracao_motor_mm_s"],
    errors="coerce"
    ).astype(float)


    df["str_consumo_energia_kwh"] = pd.to_numeric(
        df["str_consumo_energia_kwh"],
        errors="coerce"
    ).astype(float)


    df["str_data_registro"] = (
        df["str_data_registro"]
        .apply(padronizar_data)
    )

    df["ts_data_registro_pad"] = pd.to_datetime(
        df["str_data_registro"],
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce"
    )

    df = df.drop(columns=["str_data_registro"])

    print(f"\n{'='*80}")
    print("---------Padronizando unidades de medida-------------")
    print(f"{'='*80}")

    # TODO 4: criar temperatura_c

    df["dec_temperatura_c"] = np.where(
        df["str_unidade_temperatura"] == "F",
        (df["dec_temperatura"] - 32) * 5 / 9,
        df["dec_temperatura"]
    )

    # TODO 5: criar pressao_bar

    df["dec_pressao_bar"] = np.where(
        df["str_unidade_pressao"] == "psi",
        df["dec_pressao"] / 14.5038,
        df["dec_pressao"]
    )
    # TODO 6: verificar as primeiras linhas após as conversões
    print(f"\n{'='*80}")
    print("---------Padronizando colunas e limpeza-------------")
    print(f"{'='*80}")

    df = df.drop(columns=[
        "dec_temperatura",
        "dec_pressao",
        "str_unidade_temperatura",
        "str_unidade_pressao"
    ])
    
    return df

def categorical_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans categorical data in the dataset.
    """

    print(f"\n{'='*80}")
    print("---------Padronizando Categorias-------------")
    print(f"{'='*80}")
    # TODO 1: criar dicionários de padronização para linha_producao e turno

    mapa_linha = {
        "L-A":"A",
        "L-B":"B",
        "L-C":"C",
        "Linha A": "A",
        "Linha B": "B",
        "Linha C": "C",
        "linha a": "A",
        "linha b": "B",
        "linha c": "C",
        "linha_A": "A",
        "linha_B": "B",
        "linha_C": "C",
        "?": np.nan
    }

    mapa_turno = {
        "M": "manha",
        "MANHA": "manha",
        "Manhã": "manha",
        "Matutino": "manha",
        "N": "noite",
        "NOITE": "noite",
        "Noite": "noite",
        "Noturno": "noite",
        "T": "tarde",
        "TARDE": "tarde",
        "Tarde": "tarde",
        "Vespertino": "tarde",
        "sem_info": np.nan
    }

    # TODO 2: aplicar os mapas

    # Corrigir linha_producao
    df["str_linha_producao_pad"] = (
        df["str_linha_producao"]
        .replace(mapa_linha)
    )

    # Corrigir turno
    df["str_turno_pad"] = (
        df["str_turno"]
        .replace(mapa_turno)
    )

    # Corrigir Maquina
    df["str_maquina"] = df["str_maquina"].replace(
        "M-erro",
        "desconhecido"
    )

    # TODO 3: tratar valores não mapeados
    colunas = ["str_linha_producao_pad", "str_turno_pad"]

    for col in colunas:
        moda = df[col].mode().iloc[0]
        df[col] = df[col].fillna(moda)

        print(f"{col}: NaN substituídos pela moda '{moda}'")



    df = df.drop(columns=[
        "str_turno",
        "str_linha_producao"
    ])

    df = df[
        df["str_maquina"] != "desconhecido"
    ]

    return df

def data_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    """

    # ------------------------------------------------------------
    # Atividade 4 — Limpeza de duplicatas, valores impossíveis e ausentes
    # ------------------------------------------------------------

    # TODO 1: remover duplicatas
    df = df.drop_duplicates()
    # TODO 4: verificar duplicatas
    n_duplicatas  = df.duplicated().sum()
    print(f"\n{'='*80}")
    print("---------Registros Duplicados Após tratamento-------------")
    print(f"{'='*80}")
    print("Linhas idênticas: ", n_duplicatas)
    n_duplicatas  = df['str_registro_id'].duplicated().sum()
    print("ID repetipo: ", n_duplicatas)


    # TODO 2: criar regras para valores fisicamente impossíveis
    # Exemplos de regras possíveis:
    # - temperatura_c entre 500 e 950
    # - pressao_bar entre 0.5 e 12
    # - vibracao_motor_mm_s >= 0
    # - umidade_pct entre 0 e 100
    # - velocidade_esteira_m_min > 0
    # - tamanho_lote > 0



    # TODO 5: verificar estatísticas descritivas das variáveis numéricas
    

    # ============================================================
    # VALIDAÇÃO DE PLAUSIBILIDADE FÍSICA DOS DADOS
    # ============================================================

    # ============================================================
    # Definição das regras
    # ============================================================

    regras = {

        # Idade da máquina
        "big_idade_maquina_anos": (
            (df["big_idade_maquina_anos"] >= 0) &
            (df["big_idade_maquina_anos"] <= 100),
            "0 <= idade_maquina_anos <= 100"
        ),

        # Temperatura
        "dec_temperatura_c": (
            (
                (
                df["dec_temperatura_c"].between(-273.15, 2000))
            ),
            "Temperatura acima do zero absoluto"
        ),

        # Pressão
        "dec_pressao_bar": (
            (df["dec_pressao_bar"] >= 0),
            "pressao_bar >= 0"
        ),

        # Vibração
        "str_vibracao_motor_mm_s": (
            (df["str_vibracao_motor_mm_s"] >= 0),
            "0 <= vibracao_motor_mm_s"
        ),

        # Velocidade da esteira
        "dec_velocidade_esteira_m_min": (
            (df["dec_velocidade_esteira_m_min"] >= 0) ,
            "0 <= velocidade_esteira_m_min"
        ),

        # Umidade
        "dec_umidade_pct": (
            (df["dec_umidade_pct"] >= 0) &
            (df["dec_umidade_pct"] <= 100),
            "0 <= umidade_pct <= 100"
        ),

        # Tamanho do lote
        "big_tamanho_lote": (
            (df["big_tamanho_lote"] > 0),
            "tamanho_lote > 0"
        ),


        # Taxa de defeitos
        "dec_taxa_defeitos_pct": (
            (df["dec_taxa_defeitos_pct"] >= 0) &
            (df["dec_taxa_defeitos_pct"] <= 100),
            "0 <= taxa_defeitos_pct <= 100"
        ),

        # Energia sensor
        "dec_energia_sensor_b_kwh": (
            (df["dec_energia_sensor_b_kwh"] >= 0),
            "energia_sensor_b_kwh >= 0"
        ),

        # Consumo energia
        "str_consumo_energia_kwh": (
            (df["str_consumo_energia_kwh"] >= 0),
            "consumo_energia_kwh >= 0"
        ),

        # Falha binária
        "big_falha_24h": (
            df["big_falha_24h"].isin([0, 1]),
            "falha_24h deve ser 0 ou 1"
        )
    }

    # ============================================================
    # Resumo consolidado
    # ============================================================

    resumo = []

    for nome, (condicao, descricao) in regras.items():
        resumo.append([
            nome,
            descricao,
            int((~condicao).sum())
        ])


    resumo_df = pd.DataFrame(
        resumo,
        columns=["Regra", "Descrição", "Qtd_Violações"]
    )



    # ============================================================
    # Substituir valores fisicamente impossíveis por NaN
    # ============================================================

    for coluna, (condicao, descricao) in regras.items():
        qtd = (~condicao).sum()

        if qtd > 0:
            df.loc[~condicao, coluna] = np.nan

            print(
                f"{coluna}: {qtd} valor(es) inválido(s) "
                f"substituído(s) por NaN"
            )

 


    # TODO 3: substituir valores fora dos limites por NaN
    print(f"\n{'='*80}")
    print("---------Substituição de valores ausentes-------------")
    print(f"{'='*80}")


    # Lista das colunas numéricas
    colunas = [
        "str_vibracao_motor_mm_s",
        "str_consumo_energia_kwh",
        "dec_umidade_pct",
        "dec_pressao_bar",
        "dec_velocidade_esteira_m_min",
        "dec_temperatura_c"
    ]

    def check_distribuicao(df,colunas):

        resumo = []

        for col in colunas:

            serie = df[col].dropna()

            # Percentual de ausentes
            qtd_nan = df[col].isna().sum()
            pct_nan = 100 * qtd_nan / len(df)

            # Assimetria
            skew = serie.skew()

            # Outliers pelo método IQR
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1

            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            qtd_outliers = (
                (serie < limite_inf) |
                (serie > limite_sup)
            ).sum()

            pct_outliers = 100 * qtd_outliers / len(serie)

            # Interpretação da assimetria
            if abs(skew) < 0.5:
                distribuicao = "Aproximadamente simétrica"
            elif abs(skew) < 1:
                distribuicao = "Moderadamente assimétrica"
            else:
                distribuicao = "Fortemente assimétrica"

            # Sugestão de imputação
            if abs(skew) < 0.5 and pct_outliers < 5:
                sugestao = "Média"
            else:
                sugestao = "Mediana"

            resumo.append({
                "Variável": col,
                "NaN": qtd_nan,
                "% NaN": round(pct_nan, 2),
                "Distribuição": distribuicao,
                "Outliers": qtd_outliers,
                "% Outliers": round(pct_outliers, 2),
                "Sugestão Imputação": sugestao
            })

        resumo_df = pd.DataFrame(resumo)

        print(resumo_df)

    check_distribuicao(df,colunas)

    colunas = [
        "str_vibracao_motor_mm_s",
        "str_consumo_energia_kwh",
        "dec_temperatura_c"
    ]

    for col in colunas:
        df[col] = df[col].fillna(
            df[col].median()
        )

    colunas = [
        "dec_umidade_pct",
        "dec_pressao_bar",
        "dec_velocidade_esteira_m_min"
    ]

    for col in colunas:
        df[col] = df[col].fillna(
            df[col].mean()
        )

    print("Devido assimetria na distribuição será aplicado mediana nas variaveis: vibracao_motor_mm_s, consumo_energia_kwh e temperatura_c")
    print("Devido simetria na distribuição será aplicado média nas variaveis: umidade_pct   , pressao_bar    e velocidade_esteira_m_min   ")

    # TODO 4: avaliar novamente valores ausentes
    print("Devido a baixa quantidade de registros com ausencia de valor na variavel tamanho_lote, será conduzido a remoção dos registros")


    # TODO 5: decidir quais linhas remover, se necessário
    df = df[
        df["big_tamanho_lote"] > 0
    ]


    return df

def main(df=None, table_name=None, spark_session = None) -> any:
    """
    Main function to preprocess the dataset.
    """
    if df is None:
        df = spark.table("datapulse.bronze."+table_name)
    
    if isinstance(df, SparkDataFrame):
        df = df.toPandas()

    # Typing data
    df = typing_data(df, table_name)

    # Categorical data cleaning
    df = categorical_data_cleaning(df)

    # Handle missing values
    df = data_missing_values(df)

    df = corrigir_schema(df)

    if spark_session is None:
        df_spark = spark.createDataFrame(df)
    else:
        df_spark = spark_session.createDataFrame(df)

    df_spark.write.format("delta").mode("overwrite").saveAsTable("datapulse.silver."+table_name)

    return df_spark


if __name__ == "__main__":
    main(None, table_name = "sensor_events")