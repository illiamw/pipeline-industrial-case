# Databricks Pipeline Template

Template para organizar um pipeline de dados no Databricks usando as camadas Bronze, Silver e Gold. Os arquivos Python, configurações e testes incluídos são pontos de partida; implemente a lógica de negócio antes de executar o pipeline em produção.

## Estrutura

- `notebooks/`: pontos de entrada das tarefas de ingestão, transformação e manutenção.
- `src/data_pipeline/`: módulos reutilizáveis de ingestão, transformação, validação, armazenamento e utilitários.
- `resources/`: definições de jobs Databricks em YAML, incluídas pelo `databricks.yml`.
- `jobs/`: exemplos JSON dos jobs; não são carregados pelo bundle atual.
- `configs/`: arquivos de configuração por ambiente.
- `tests/`: testes automatizados para os módulos em `src/data_pipeline/`.
- `docs/`: documentação de arquitetura, dados e operação.

## Jobs Databricks

O bundle usa os arquivos `resources/*.job.yml` como fonte de verdade para criação dos jobs. Há um arquivo para cada camada: `ingestion.job.yml`, `bronze.job.yml`, `silver.job.yml` e `gold.job.yml`.

Ao adicionar ou alterar um job:

1. Defina-o em `resources/<nome>.job.yml`, dentro de `resources.jobs`.
2. Use uma chave de job estável e um `name` claro. Cada `task_key` deve ser único dentro do job.
3. Aponte `notebook_path` para um arquivo existente em `notebooks/` e mantenha `source: WORKSPACE` para os notebooks deste template.
4. Configure compute, parâmetros e permissões conforme o workspace e as necessidades da tarefa.
5. Se uma tarefa depender de outra, declare `depends_on` explicitamente. Sem dependências, as tarefas de um job podem executar em paralelo.
6. Valide o bundle antes do deploy.

Os arquivos `jobs/*.json` são exemplos separados: o `databricks.yml` não os inclui nem os usa para deploy. O workflow de CI apenas verifica se esses arquivos contêm JSON válido. Para mudar o job implantado pelo bundle, edite o YAML correspondente em `resources/`.

## Pré-requisitos

- Python 3.11 ou compatível com as dependências do projeto.
- Databricks CLI instalado e autenticado para o workspace de destino.
- Acesso e compute configurados no workspace para executar as tarefas.

Autentique-se localmente com o perfil desejado, por exemplo:

```powershell
databricks auth login --host "https://<workspace-url>"
```

Substitua o endereço de exemplo pelo URL real do workspace. Não armazene tokens ou outros segredos no repositório; use os mecanismos de autenticação do Databricks CLI ou o gerenciador de segredos da sua plataforma.

## Validar e executar

O alvo `dev` é o padrão. O endereço do workspace é uma variável do bundle chamada `databricks_host`:

```powershell
databricks bundle validate -t dev --var="databricks_host=https://<workspace-url>"
databricks bundle deploy -t dev --var="databricks_host=https://<workspace-url>"
databricks bundle run -t dev ingestion_job
```

Troque `ingestion_job` pelo nome do job definido em `resources/` para executar outra camada. Para produção, use `-t prod` após validar as configurações, permissões e dados do ambiente de destino.

## Testes

Implemente os casos de teste em `tests/`, usando nomes reconhecidos pelo pytest, como `test_*.py` e funções `test_*`. Teste as funções de `src/data_pipeline/` com dados pequenos e isolados; evite depender de um workspace Databricks nos testes unitários.

Para executar localmente, instale o pytest e rode:

```powershell
python -m pip install pytest
python -m pytest tests/
```

No estado atual, os arquivos em `tests/` são apenas esqueletos e ainda não contêm casos executáveis. O workflow `.github/workflows/ci_cd.yml` também espera `requirements-dev.txt`, executa Ruff e exige cobertura mínima de 80%; crie esse arquivo com as ferramentas de desenvolvimento e implemente testes suficientes antes de depender do CI como gate de deploy.

Adicione as dependências de runtime usadas pelo pipeline a `requirements.txt` e as ferramentas de desenvolvimento a `requirements-dev.txt`. Não inclua credenciais nesses arquivos.
