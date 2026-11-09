import pandas as pd
import os
# Bibliotecas básicas
import numpy as np

RANDOM_STATE = 42



def gerate_data(file_path: str) -> pd.DataFrame:
    """
    Gerates a pandas DataFrame from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # ------------------------------------------------------------
    # Geração de uma base sintética industrial com inconsistências
    # ------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_STATE)
    n = 10000

    # Variáveis estruturais limpas, usadas internamente para gerar os dados
    datas = pd.date_range("2025-01-01 00:00:00", periods=n, freq="6h")

    linha_limpa = rng.choice(["A", "B", "C"], size=n, p=[0.42, 0.34, 0.24])
    turno_limpo = rng.choice(["manha", "tarde", "noite"], size=n, p=[0.34, 0.33, 0.33])
    maquina_limpa = rng.choice([f"M{i:02d}" for i in range(1, 9)], size=n)

    efeito_linha_temp = np.select(
        [linha_limpa == "A", linha_limpa == "B", linha_limpa == "C"],
        [0, 18, -12]
    )

    efeito_turno_temp = np.select(
        [turno_limpo == "manha", turno_limpo == "tarde", turno_limpo == "noite"],
        [-5, 8, 3]
    )

    idade_maquina_anos = rng.integers(1, 16, size=n)
    temperatura_c_real = 760 + efeito_linha_temp + efeito_turno_temp + rng.normal(0, 18, size=n)
    pressao_bar_real = 5.5 + 0.03 * (temperatura_c_real - 760) + rng.normal(0, 0.35, size=n)
    vibracao_mm_s_real = 1.5 + 0.13 * idade_maquina_anos + rng.normal(0, 0.45, size=n)
    velocidade_esteira_real = 42 + rng.normal(0, 4.5, size=n)
    umidade_pct_real = np.clip(55 + rng.normal(0, 12, size=n), 18, 96)
    tamanho_lote_real = rng.integers(800, 5000, size=n)
    tempo_setup_min_real = np.clip(35 + 1.2 * idade_maquina_anos + rng.normal(0, 12, size=n), 5, 120)
    paradas_nao_planejadas_real = rng.poisson(lam=np.clip(0.3 + 0.06 * idade_maquina_anos, 0.2, 2.0), size=n)

    # Alvo de regressão: consumo de energia
    consumo_energia_real = (
        210
        + 0.18 * temperatura_c_real
        + 5.8 * pressao_bar_real
        + 7.0 * vibracao_mm_s_real
        + 0.015 * tamanho_lote_real
        + 1.4 * tempo_setup_min_real
        + 14 * paradas_nao_planejadas_real
        + rng.normal(0, 22, size=n)
    )

    # Alvo auxiliar de qualidade
    taxa_defeitos_pct_real = np.clip(
        1.5
        + 0.035 * np.abs(temperatura_c_real - 765)
        + 0.42 * np.maximum(vibracao_mm_s_real - 2.5, 0)
        + 0.12 * paradas_nao_planejadas_real
        + rng.normal(0, 0.7, size=n),
        0,
        18
    )

    # Alvo de classificação: falha operacional em até 24h
    logit_falha = (
        -5.0
        + 0.020 * (temperatura_c_real - 760)
        + 0.55 * np.maximum(vibracao_mm_s_real - 2.2, 0)
        + 0.22 * paradas_nao_planejadas_real
        + 0.018 * tempo_setup_min_real
        + 0.065 * idade_maquina_anos
    )

    prob_falha = 1 / (1 + np.exp(-logit_falha))
    falha_24h = rng.binomial(1, prob_falha, size=n)

    # Formatos mistos de datas
    formatos_data = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%m-%d-%Y %H:%M"]
    data_registro = [
        dt.strftime(formatos_data[i % len(formatos_data)])
        for i, dt in enumerate(datas)
    ]

    # Inconsistências categóricas
    aliases_linha = {
        "A": ["Linha A", "linha a", "A", "L-A", "linha_A"],
        "B": ["Linha B", "linha b", "B", "L-B", "linha_B"],
        "C": ["Linha C", "linha c", "C", "L-C", "linha_C"],
    }

    aliases_turno = {
        "manha": ["Manhã", "MANHA", "manha", "Matutino", "M"],
        "tarde": ["Tarde", "TARDE", "tarde", "Vespertino", "T"],
        "noite": ["Noite", "NOITE", "noite", "Noturno", "N"],
    }

    linha_raw = [rng.choice(aliases_linha[x]) for x in linha_limpa]
    turno_raw = [rng.choice(aliases_turno[x]) for x in turno_limpo]

    # Mistura de unidades: Celsius/Fahrenheit e bar/psi
    usa_fahrenheit = rng.random(n) < 0.18
    temperatura_valor = np.where(
        usa_fahrenheit,
        temperatura_c_real * 9 / 5 + 32,
        temperatura_c_real
    )
    unidade_temperatura = np.where(usa_fahrenheit, "F", "C")

    usa_psi = rng.random(n) < 0.22
    pressao_valor = np.where(
        usa_psi,
        pressao_bar_real * 14.5038,
        pressao_bar_real
    )
    unidade_pressao = np.where(usa_psi, "psi", "bar")

    # Montagem do DataFrame bruto
    dados_raw = pd.DataFrame({
        "str_registro_id": [f"REG-{i:05d}" for i in range(n)],
        "str_data_registro": data_registro,
        "str_linha_producao": linha_raw,
        "str_turno": turno_raw,
        "str_maquina": maquina_limpa,
        "big_idade_maquina_anos": idade_maquina_anos,
        "dec_temperatura": np.round(temperatura_valor, 2),
        "str_unidade_temperatura": unidade_temperatura,
        "dec_pressao": np.round(pressao_valor, 3),
        "str_unidade_pressao": unidade_pressao,
        "str_vibracao_motor_mm_s": np.round(vibracao_mm_s_real, 3),
        "dec_velocidade_esteira_m_min": np.round(velocidade_esteira_real, 2),
        "dec_umidade_pct": np.round(umidade_pct_real, 2),
        "big_tamanho_lote": tamanho_lote_real,
        "dec_tempo_setup_min": np.round(tempo_setup_min_real, 2),
        "big_paradas_nao_planejadas": paradas_nao_planejadas_real,
        "dec_taxa_defeitos_pct": np.round(taxa_defeitos_pct_real, 3),
        "dec_energia_sensor_b_kwh": np.round(consumo_energia_real * 1.015 + rng.normal(0, 8, size=n), 2),
        "str_codigo_campanha": rng.choice(["A12", "B07", "C99", "X00"], size=n),
        "dec_ruido_aleatorio": rng.normal(0, 1, size=n),
        "str_consumo_energia_kwh": np.round(consumo_energia_real, 2),
        "big_falha_24h": falha_24h
    })

    # Converter colunas selecionadas para object antes de inserir textos com vírgula decimal
    dados_raw["str_consumo_energia_kwh"] = dados_raw["str_consumo_energia_kwh"].astype(object)
    dados_raw["str_vibracao_motor_mm_s"] = dados_raw["str_vibracao_motor_mm_s"].astype(object)

    # Transformar parte dos números em textos com vírgula decimal
    idx_consumo_texto = rng.choice(dados_raw.index, size=80, replace=False)
    dados_raw.loc[idx_consumo_texto, "str_consumo_energia_kwh"] = (
        dados_raw.loc[idx_consumo_texto, "str_consumo_energia_kwh"]
        .map(lambda x: f"{float(x):.2f}".replace(".", ","))
    )

    idx_vibracao_texto = rng.choice(dados_raw.index, size=55, replace=False)
    dados_raw.loc[idx_vibracao_texto, "str_vibracao_motor_mm_s"] = (
        dados_raw.loc[idx_vibracao_texto, "str_vibracao_motor_mm_s"]
        .map(lambda x: f"{float(x):.3f}".replace(".", ","))
    )

    # Inserir categorias inválidas
    idx_cat = rng.choice(dados_raw.index, size=30, replace=False)
    dados_raw.loc[idx_cat[:10], "str_linha_producao"] = "?"
    dados_raw.loc[idx_cat[10:20], "str_turno"] = "sem_info"
    dados_raw.loc[idx_cat[20:], "str_maquina"] = "M-erro"

    # Inserir valores ausentes
    cols_missing = [
        "dec_temperatura", "dec_pressao", "str_vibracao_motor_mm_s",
        "dec_velocidade_esteira_m_min", "dec_umidade_pct", "str_turno", "str_linha_producao"
    ]
    for col in cols_missing:
        idx_nan = rng.choice(dados_raw.index, size=int(0.035 * n), replace=False)
        dados_raw.loc[idx_nan, col] = np.nan

    # Inserir valores fisicamente impossíveis
    idx_bad = rng.choice(dados_raw.index, size=32, replace=False)
    dados_raw.loc[idx_bad[:6], "dec_temperatura"] = -50
    dados_raw.loc[idx_bad[6:12], "dec_pressao"] = -3
    dados_raw.loc[idx_bad[12:18], "str_vibracao_motor_mm_s"] = -1
    dados_raw.loc[idx_bad[18:24], "dec_umidade_pct"] = 140
    dados_raw.loc[idx_bad[24:28], "dec_velocidade_esteira_m_min"] = -20
    dados_raw.loc[idx_bad[28:], "big_tamanho_lote"] = 0

    # Inserir outliers extremos plausíveis ou suspeitos
    idx_out = rng.choice(dados_raw.index, size=18, replace=False)
    dados_raw.loc[idx_out[:6], "str_consumo_energia_kwh"] = (
        pd.to_numeric(
            dados_raw.loc[idx_out[:6], "str_consumo_energia_kwh"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        ) * 2.7
    )
    dados_raw.loc[idx_out[6:12], "str_vibracao_motor_mm_s"] = 12 + rng.normal(0, 1, size=6)
    dados_raw.loc[idx_out[12:], "dec_tempo_setup_min"] = 240 + rng.normal(0, 10, size=6)

    # Inserir alguns registros duplicados
    duplicatas = dados_raw.sample(35, random_state=RANDOM_STATE)
    dados_raw = pd.concat([dados_raw, duplicatas], ignore_index=True)

    # Garantir tipo consistente para colunas mistas (string + número) antes da conversão Arrow
    dados_raw["str_consumo_energia_kwh"] = dados_raw["str_consumo_energia_kwh"].astype(str)
    dados_raw["str_vibracao_motor_mm_s"] = dados_raw["str_vibracao_motor_mm_s"].astype(str)

    df_spark = spark.createDataFrame(dados_raw)

    df_spark.write.format("delta").mode("overwrite").saveAsTable(file_path)


def main(file_path):
    """
    Main function to generate synthetic industrial data.
    """
    
    gerate_data(file_path)
    print(f"Data generated and saved to {file_path}")


if __name__ == "__main__":
    main(file_path = "datapulse.bronze.sensor_events")