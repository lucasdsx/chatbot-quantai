import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

# Esta é a função que será chamada pelo chatbot.py
def run_backtest(
    uploaded_file, 
    col_data, 
    col_ticker,
    col_fator_qualidade, # Novo
    col_fator_valor,     # Novo
    col_fator_risco,     # Novo
    n_portfolio,
    taxa_livre_risco_anual # Novo
    ):
    
    # -------------------------------------------------------------------------
    # CONFIGURAÇÃO (Valores fixos ou recebidos)
    # -------------------------------------------------------------------------
    NOME_DO_ARQUIVO = uploaded_file.name
    
    # Mapeamento de colunas (recebido)
    COLUNA_DE_DATA = col_data
    COLUNA_DE_TICKER = col_ticker
    COL_FATOR_Q = col_fator_qualidade
    COL_FATOR_V = col_fator_valor
    COL_FATOR_R = col_fator_risco

    # Parâmetros da Estratégia (recebido)
    N_PORTFOLIO = n_portfolio
    TAXA_LIVRE_RISCO_ANUAL = taxa_livre_risco_anual
    
    # Parâmetros Fixos (da lógica do backtest2.py)
    CUSTO_POR_OPERACAO = 0.0015
    MESES_REBALANCEAMENTO = [1, 4, 7, 10]
    EXCLUIR_EV_EBITDA_NAO_POSITIVO = True
    FILTRAR_LIQUIDEZ = True
    LIMIAR_MIN_PREGOES_VALIDOS = 50
    
    # Nomes internos para os fatores, para garantir consistência
    FATOR_Q_INTERNO = 'FATOR_Q'
    FATOR_V_INTERNO = 'FATOR_V'
    FATOR_R_INTERNO = 'FATOR_R'
    
    # -------------------------------------------------------------------------
    # CÉLULA 2: CARREGAR DADOS
    # -------------------------------------------------------------------------
    print(f"Carregando arquivo: {NOME_DO_ARQUIVO}...")
    try:
        # Determina o tipo de arquivo
        if NOME_DO_ARQUIVO.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif NOME_DO_ARQUIVO.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError("Formato de arquivo não suportado. Use .xlsx ou .csv")
            
        df[COLUNA_DE_DATA] = pd.to_datetime(df[COLUNA_DE_DATA])
        df['Ano'] = df[COLUNA_DE_DATA].dt.year
        df['Trimestre'] = df[COLUNA_DE_DATA].dt.quarter
        df['AnoTrimestre'] = df['Ano'].astype(str) + '-Q' + df['Trimestre'].astype(str)
        print("Arquivo carregado e colunas de trimestre criadas.")
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}. Verifique os nomes das colunas.")
        raise
        
    # -------------------------------------------------------------------------
    # CÉLULA 3: LIMPEZA E AMOSTRAGEM
    # -------------------------------------------------------------------------
    print("Limpando e amostrando dados...")
    
    # Renomeia colunas do usuário para nomes fixos internos
    colunas_fatores_originais = [COL_FATOR_Q, COL_FATOR_V, COL_FATOR_R]
    colunas_fatores_internos = [FATOR_Q_INTERNO, FATOR_V_INTERNO, FATOR_R_INTERNO]
    
    rename_map = {
        COLUNA_DE_TICKER: 'Ticker',
        **dict(zip(colunas_fatores_originais, colunas_fatores_internos))
    }
    
    # Verifica se todas as colunas necessárias existem
    colunas_necessarias = [COLUNA_DE_DATA, COLUNA_DE_TICKER, COL_FATOR_Q, COL_FATOR_V, COL_FATOR_R]
    if not all(col in df.columns for col in colunas_necessarias):
        colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]
        raise ValueError(f"Colunas não encontradas no arquivo: {colunas_faltando}. Verifique o mapeamento na barra lateral.")
        
    df = df.rename(columns=rename_map)
    
    df['Prefixo'] = df['Ticker'].str.slice(0, 4)

    # Remove ON quando já existe PN (heurística)
    pn_tickers = df[df['Ticker'].str.endswith('4')]['Prefixo'].unique()
    df = df[~((df['Prefixo'].isin(pn_tickers)) & (df['Ticker'].str.endswith('3')))]

    df_limpo_diario = df.dropna(subset=colunas_fatores_internos)
    df_ordenado = df_limpo_diario.sort_values(by=['Ticker', COLUNA_DE_DATA])

    cont_pregoes = (
        df_ordenado.groupby(['Ticker', 'AnoTrimestre'])[COLUNA_DE_DATA]
        .nunique()
        .reset_index(name='Pregoes')
    )

    df_trimestral = df_ordenado.groupby(['Ticker', 'AnoTrimestre']).last().reset_index()
    df_trimestral = df_trimestral.merge(cont_pregoes, on=['Ticker', 'AnoTrimestre'], how='left')

    # LAG de disponibilidade
    df_trimestral = df_trimestral.sort_values(['Ticker', 'AnoTrimestre'])
    for col in colunas_fatores_internos:
        df_trimestral[f'{col}_LAG1'] = df_trimestral.groupby('Ticker')[col].shift(1)
    df_trimestral['Pregoes_LAG1'] = df_trimestral.groupby('Ticker')['Pregoes'].shift(1)

    lag_cols = [f'{c}_LAG1' for c in colunas_fatores_internos]
    df_trimestral_lag = df_trimestral.dropna(subset=lag_cols).copy()

    if FILTRAR_LIQUIDEZ:
        df_trimestral_lag = df_trimestral_lag[
            (df_trimestral_lag['Pregoes_LAG1'].fillna(0) >= LIMIAR_MIN_PREGOES_VALIDOS)
        ].copy()

    print(f"Dados amostrados por trimestre com LAG: {len(df_trimestral_lag)} linhas")

    # -------------------------------------------------------------------------
    # CÉLULA 4: FUNÇÃO DE RANKING (Z-SCORE)
    # -------------------------------------------------------------------------
    # Nomes internos com _LAG1
    FATOR_Q_LAG = f"{FATOR_Q_INTERNO}_LAG1"
    FATOR_V_LAG = f"{FATOR_V_INTERNO}_LAG1"
    FATOR_R_LAG = f"{FATOR_R_INTERNO}_LAG1"

    def calcular_master_rank(df_periodo):
        dfp = df_periodo.copy()

        if EXCLUIR_EV_EBITDA_NAO_POSITIVO:
            # Assumindo que o FATOR_V é o EV/EBITDA como na lógica original
            # Se o usuário mapear outro fator, esta lógica pode precisar de ajuste
            # Mas mantendo a premissa do backtest2.py:
            dfp = dfp[dfp[FATOR_V_LAG] > 0].copy()

        if len(dfp) <= N_PORTFOLIO:
            dfp['Master_Score'] = 0.0
            return dfp.sort_values(by=['Ticker'])

        # Calcular Z-Scores dos fatores defasados (LAG1)
        dfp['z_qualidade'] = (
            dfp[FATOR_Q_LAG].rank().sub(dfp[FATOR_Q_LAG].rank().mean())
        ).div(dfp[FATOR_Q_LAG].rank().std())
        
        dfp['z_valor'] = (
            dfp[FATOR_V_LAG].rank().sub(dfp[FATOR_V_LAG].rank().mean())
        ).div(dfp[FATOR_V_LAG].rank().std())
        
        dfp['z_risco'] = (
            dfp[FATOR_R_LAG].rank().sub(dfp[FATOR_R_LAG].rank().mean())
        ).div(dfp[FATOR_R_LAG].rank().std())

        # Combina Z-Scores (Queremos: Qualidade alta, Valor baixo, Risco baixo)
        dfp['Master_Score'] = dfp['z_qualidade'] - dfp['z_valor'] - dfp['z_risco']
        
        return dfp.sort_values(by='Master_Score', ascending=False)

    # -------------------------------------------------------------------------
    # CÉLULA 6: BAIXANDO DADOS DE PREÇOS
    # -------------------------------------------------------------------------
    print("\nBaixando dados de preços do yfinance...")
    tickers_unicos = df_trimestral_lag['Ticker'].unique()
    
    if len(tickers_unicos) == 0:
        raise ValueError("Nenhum ticker restou após os filtros de limpeza e LAG. Verifique seu arquivo de entrada.")
        
    tickers_sa = [f"{t}.SA" for t in tickers_unicos]
    tickers_completos = tickers_sa + ['^BVSP']

    # Oculta o progresso do yfinance no Streamlit
    px_all = yf.download(tickers_completos, start='2012-01-01', end='2024-12-31', interval='1mo', auto_adjust=True, progress=False)['Close']
    px_all = px_all.dropna(how='all')
    
    if px_all.empty:
        raise ValueError("Falha ao baixar dados do yfinance. Verifique a conexão ou os tickers.")

    retornos_mensais_base = px_all.pct_change()
    retornos_benchmark_base = retornos_mensais_base['^BVSP'].dropna()
    retornos_setor_ew_base = retornos_mensais_base[tickers_sa].mean(axis=1).dropna()

    print("Dados de preços baixados.")

    valid_tickers_sa = px_all[tickers_sa].columns[px_all[tickers_sa].notna().any()].to_list()
    valid_tickers_base = [t.replace('.SA', '') for t in valid_tickers_sa]
    print(f"Filtro de dados: {len(valid_tickers_base)} de {len(tickers_sa)} tickers possuem dados de preço válidos.")

    # -------------------------------------------------------------------------
    # CÉLULA 5: EXECUTANDO O BACKTEST
    # -------------------------------------------------------------------------
    print("Executando lógica do backtest trimestral...")
    portfolio_trimestral = []
    trimestres_disponiveis = sorted(df_trimestral_lag['AnoTrimestre'].unique())

    for i in range(1, len(trimestres_disponiveis)):
        trimestre_posse = trimestres_disponiveis[i]
        trimestre_decisao = trimestres_disponiveis[i-1]

        df_decisao_bruto = df_trimestral_lag[df_trimestral_lag['AnoTrimestre'] == trimestre_decisao]
        df_decisao_filtrado = df_decisao_bruto[
            df_decisao_bruto['Ticker'].isin(valid_tickers_base)
        ]
        
        # Passa N_PORTFOLIO para a função de rank
        ranking_do_trimestre = calcular_master_rank(df_decisao_filtrado)
        top_n_tickers = ranking_do_trimestre.head(N_PORTFOLIO)['Ticker'].tolist()
        
        portfolio_trimestral.append({
            'AnoTrimestre_Posse': trimestre_posse,
            'Tickers_Selecionados': top_n_tickers
        })

    df_portfolio = pd.DataFrame(portfolio_trimestral)
    
    if df_portfolio.empty:
        raise ValueError("Nenhum portfólio pôde ser formado. Verifique o período dos seus dados e se há sobreposição suficiente com os dados de preço.")


    # -------------------------------------------------------------------------
    # CÉLULA 7: CALCULANDO RETORNOS LÍQUIDOS (LÓGICA CORRETA DO backtest2.py)
    # -------------------------------------------------------------------------
    print("Calculando retornos da estratégia (com pesos flutuantes)...")
    ret_m = retornos_mensais_base.copy()
    ret_m['Ano'] = ret_m.index.year
    ret_m['Trimestre'] = ret_m.index.quarter
    ret_m['AnoTrimestre_Posse'] = ret_m['Ano'].astype(str) + '-Q' + ret_m['Trimestre'].astype(str)
    ret_m['Eh_Mes_Rebalanceamento'] = ret_m.index.month.isin(MESES_REBALANCEAMENTO)

    df_portfolio_idx = df_portfolio.set_index('AnoTrimestre_Posse')
    df_backtest = ret_m.join(df_portfolio_idx, on='AnoTrimestre_Posse')

    primeiro_trimestre_posse = df_portfolio['AnoTrimestre_Posse'].min()
    df_backtest_final = df_backtest[df_backtest['AnoTrimestre_Posse'] >= primeiro_trimestre_posse].copy()

    # --- LÓGICA DE PESOS FLUTUANTES (do backtest2.py) ---
    pesos_no_inicio_do_mes = {}
    resultados_mensais = []

    for index, row in df_backtest_final.iterrows():
        eh_rebalanceamento = row['Eh_Mes_Rebalanceamento']
        ativos_selecionados_trimestre = row.get('Tickers_Selecionados', [])
        ativos_selecionados_set = set(ativos_selecionados_trimestre)
        pesos_atuais_para_calculo = {}
        turnover = 0.0
        custo = 0.0

        if eh_rebalanceamento:
            # 1. MÊS DE REBALANCEAMENTO
            n_ativos_validos = len(ativos_selecionados_set.intersection(valid_tickers_base))
            if n_ativos_validos > 0:
                peso_ew = 1.0 / n_ativos_validos
                pesos_alvo = {t: peso_ew for t in ativos_selecionados_set.intersection(valid_tickers_base)}
            else:
                pesos_alvo = {}

            all_tickers = ativos_selecionados_set.union(set(pesos_no_inicio_do_mes.keys()))
            for t in all_tickers:
                w_prev = pesos_no_inicio_do_mes.get(t, 0.0)
                w_new  = pesos_alvo.get(t, 0.0)
                turnover += abs(w_new - w_prev)
            
            custo = turnover * CUSTO_POR_OPERACAO
            pesos_atuais_para_calculo = pesos_alvo
        else:
            # 2. MÊS INTERMEDIÁRIO
            pesos_atuais_para_calculo = pesos_no_inicio_do_mes
            turnover = 0.0
            custo = 0.0

        # 3. CALCULAR RETORNO BRUTO E PESOS FLUTUANTES
        retorno_bruto = 0.0
        pesos_no_fim_do_mes = {}
        valor_total_fim_de_mes = 0.0

        for ticker, peso_inicial in pesos_atuais_para_calculo.items():
            ticker_sa = f"{ticker}.SA"
            # Verifica se o ticker existe nos dados de preço (pode não estar no 'row')
            if ticker_sa not in row.index: continue 
                
            retorno_mensal = row.get(ticker_sa, np.nan)
            
            if pd.notna(retorno_mensal) and peso_inicial > 0:
                retorno_bruto += peso_inicial * retorno_mensal
                peso_final_nominal = peso_inicial * (1 + retorno_mensal)
                pesos_no_fim_do_mes[ticker] = peso_final_nominal
                valor_total_fim_de_mes += peso_final_nominal
            elif peso_inicial > 0:
                pesos_no_fim_do_mes[ticker] = 0.0 # Perde o valor (delistado?)

        retorno_liquido = retorno_bruto - custo
        
        # 4. ATUALIZAR ESTADO (PARA O PRÓXIMO LOOP)
        if valor_total_fim_de_mes > 0:
            pesos_no_inicio_do_mes = {
                t: w / valor_total_fim_de_mes for t, w in pesos_no_fim_do_mes.items()
            }
        else:
            pesos_no_inicio_do_mes = {} # Portfólio zerou
        
        resultados_mensais.append({
            'Date': index,
            'Retorno_Estrategia_Bruto': retorno_bruto,
            'Turnover': turnover,
            'Custo': custo,
            'Retorno_Estrategia_Liquido': retorno_liquido,
        })
    # --- FIM DA LÓGICA DE PESOS FLUTUANTES ---

    df_resultados_corrigido = pd.DataFrame(resultados_mensais).set_index('Date')

    # Remove colunas antigas/enviesadas se existirem e junta as corrigidas
    df_backtest_final = df_backtest_final.drop(
        columns=[c for c in df_backtest_final.columns if c.startswith('Retorno_') or c in ['Turnover', 'Custo', 'Eh_Mes_Rebalanceamento', 'Tickers_Selecionados_PREV', 'Tickers_Selecionados']],
        errors='ignore'
    ).join(df_resultados_corrigido)

    retornos_estrategia = df_backtest_final['Retorno_Estrategia_Liquido'].copy()
    retornos_benchmark_bruto = retornos_benchmark_base.reindex(retornos_estrategia.index)
    retornos_setor_ew_bruto = retornos_setor_ew_base.reindex(retornos_estrategia.index)

    common_idx = (
        retornos_estrategia.index
        .intersection(retornos_benchmark_bruto.dropna().index)
        .intersection(retornos_setor_ew_bruto.dropna().index)
    )
    retornos_estrategia = retornos_estrategia.reindex(common_idx)
    retornos_benchmark = retornos_benchmark_bruto.reindex(common_idx)
    retornos_setor_ew = retornos_setor_ew_bruto.reindex(common_idx)

    print("Cálculo de retornos líquidos concluído.")
    
    if retornos_estrategia.empty or retornos_benchmark.empty:
        raise ValueError("Não foi possível calcular os retornos. O período de dados do seu arquivo pode não se sobrepor ao período de preços do Ibovespa (2012-2024).")

    # -------------------------------------------------------------------------
    # CÉLULA 8: PLOTANDO O RESULTADO
    # -------------------------------------------------------------------------
    nome_est = (
        f"Top {N_PORTFOLIO} EW (Z-Score, Liq: {LIMIAR_MIN_PREGOES_VALIDOS}d, "
        f"Custo: {CUSTO_POR_OPERACAO*100:.2f}% por trans.)"
    )

    retorno_acum_estrategia = (1 + retornos_estrategia).cumprod()
    retorno_acum_benchmark = (1 + retornos_benchmark).cumprod()
    retorno_acum_setor_ew = (1 + retornos_setor_ew).cumprod()

    print(f"\nGerando gráfico para: {nome_est}")

    # Cria a figura para retornar ao Streamlit
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(retorno_acum_estrategia.index, retorno_acum_estrategia, label=f'Estratégia \"{nome_est}\"', linewidth=2)
    ax.plot(retorno_acum_benchmark.index, retorno_acum_benchmark, label='Ibovespa (^BVSP)', linestyle='--')
    ax.plot(retorno_acum_setor_ew.index, retorno_acum_setor_ew, label='Setor (Equal-Weighted)', linestyle=':')
    ax.set_title(f'Resultado Backtest (Líquido de Custos Estimados)')
    ax.set_ylabel('Retorno Acumulado (R$ 1,00)')
    ax.set_xlabel('Data')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # -------------------------------------------------------------------------
    # CÉLULA 9: MÉTRICAS DE DESEMPENHO
    # -------------------------------------------------------------------------
    anos = len(retornos_estrategia) / 12
    # Usa a taxa de risco da UI
    taxa_livre_risco_mensal = (1 + TAXA_LIVRE_RISCO_ANUAL)**(1/12) - 1

    # Métricas Estratégia
    ret_total_estrategia = retorno_acum_estrategia.iloc[-1]
    ret_anual_estrategia = (ret_total_estrategia ** (1/anos)) - 1
    vol_anual_estrategia = retornos_estrategia.std() * np.sqrt(12)
    sharpe_estrategia = (retornos_estrategia.mean() - taxa_livre_risco_mensal) / retornos_estrategia.std() * np.sqrt(12)
    running_peak_estrategia = retorno_acum_estrategia.cummax()
    drawdown_estrategia = (retorno_acum_estrategia - running_peak_estrategia) / running_peak_estrategia
    max_drawdown_estrategia = drawdown_estrategia.min()

    # Métricas Benchmark
    ret_total_benchmark = retorno_acum_benchmark.iloc[-1]
    ret_anual_benchmark = (ret_total_benchmark ** (1/anos)) - 1
    vol_anual_benchmark = retornos_benchmark.std() * np.sqrt(12)
    sharpe_benchmark = (retornos_benchmark.mean() - taxa_livre_risco_mensal) / retornos_benchmark.std() * np.sqrt(12)
    running_peak_benchmark = retorno_acum_benchmark.cummax()
    drawdown_benchmark = (retorno_acum_benchmark - running_peak_benchmark) / running_peak_benchmark
    max_drawdown_benchmark = drawdown_benchmark.min()

    # Métricas Setor EW
    ret_total_setor_ew = retorno_acum_setor_ew.iloc[-1]
    ret_anual_setor_ew = (ret_total_setor_ew ** (1/anos)) - 1
    vol_anual_setor_ew = retornos_setor_ew.std() * np.sqrt(12)
    sharpe_setor_ew = (retornos_setor_ew.mean() - taxa_livre_risco_mensal) / retornos_setor_ew.std() * np.sqrt(12)
    running_peak_setor_ew = retorno_acum_setor_ew.cummax()
    drawdown_setor_ew = (retorno_acum_setor_ew - running_peak_setor_ew) / running_peak_setor_ew
    max_drawdown_setor_ew = drawdown_setor_ew.min()

    # Cria o DataFrame de Métricas
    data = {
        'Métrica': [
            'Retorno Total (x)',
            'Retorno Anualizado (%)',
            'Volatilidade Anualizada (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)'
        ],
        f'Estratégia ({nome_est})': [
            ret_total_estrategia,
            ret_anual_estrategia * 100,
            vol_anual_estrategia * 100,
            sharpe_estrategia,
            max_drawdown_estrategia * 100
        ],
        'Ibovespa': [
            ret_total_benchmark,
            ret_anual_benchmark * 100,
            vol_anual_benchmark * 100,
            sharpe_benchmark,
            max_drawdown_benchmark * 100
        ],
        'Setor (Equal-Weighted)': [
            ret_total_setor_ew,
            ret_anual_setor_ew * 100,
            vol_anual_setor_ew * 100,
            sharpe_setor_ew,
            max_drawdown_setor_ew * 100
        ]
    }
    df_metricas = pd.DataFrame(data).set_index('Métrica')
    df_metricas_formatado = df_metricas.applymap(lambda x: f"{x:.2f}") # Formata para 2 casas
    
    # -------------------------------------------------------------------------
    # CÉLULA 11 & 12: ANÁLISE ESTATÍSTICA
    # -------------------------------------------------------------------------
    
    # --- TEXTO SIMPLES PARA O CONTEXTO DA IA ---
    analise_texto_ia = "--- MÉTRICAS DE DESEMPENHO (LÍQUIDAS) ---\n"
    # Usamos o DataFrame não formatado para a IA, para que ela veja os números
    analise_texto_ia += df_metricas.to_string(float_format="%.4f") + "\n"

    # --- MARKDOWN PARA O USUÁRIO ---
    analise_md = "### Análise Estatística (vs. Ibovespa)\n\n"

    # Teste T
    t_statistic, p_value = stats.ttest_1samp(retornos_estrategia, 0, alternative='greater')
    analise_md += f"**Teste-T (Retorno Líquido > 0)**\n"
    analise_md += f"* **P-Valor:** `{p_value:.4f}`\n"
    if p_value < 0.10:
        analise_md += "* **Conclusão:** Evidência moderada a forte de que o retorno médio líquido é positivo.\n\n"
    else:
        analise_md += "* **Conclusão:** O retorno médio líquido não é estatisticamente distinguível de zero.\n\n"
    
    analise_texto_ia += f"\n--- TESTE T (Retorno Líquido > 0) ---\n"
    analise_texto_ia += f"P-Valor: {p_value:.6f}\n"

    # Regressão Alfa/Beta
    Y = retornos_estrategia
    X = retornos_benchmark
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    alfa_coef = model.params['const']
    alfa_p_val = model.pvalues['const']
    beta_name = [c for c in model.params.index if c != 'const'][0]
    beta_coef = model.params[beta_name]
    beta_p_val = model.pvalues[beta_name]

    analise_md += f"**Regressão de Alfa (HAC Newey-West)**\n"
    analise_md += f"* **Alfa (Retorno Extra Líquido):** `{alfa_coef*12*100:.2f}%` ao ano (P-Valor: `{alfa_p_val:.4f}`)\n"
    analise_md += f"* **Beta (Exposição ao Mercado):** `{beta_coef:.4f}` (P-Valor: `{beta_p_val:.4f}`)\n"

    if alfa_p_val > 0.10 and beta_p_val < 0.10:
        analise_md += f"* **Interpretação:** Alfa não é estatisticamente significante. A estratégia parece ter capturado o risco de mercado (Beta) de forma significante.\n"
    elif alfa_p_val < 0.10:
        analise_md += "* **Interpretação:** A estratégia gerou Alfa líquido estatisticamente significante.\n"
    else:
        analise_md += "* **Interpretação:** O modelo não demonstrou significância estatística clara em Alfa ou Beta.\n"

    analise_texto_ia += f"\n--- REGRESSÃO DE ALFA (LÍQUIDO vs. Ibovespa) ---\n"
    analise_texto_ia += f"Alfa Anualizado (%): {alfa_coef*12*100:.2f} (P-Valor: {alfa_p_val:.4f})\n"
    analise_texto_ia += f"Beta: {beta_coef:.4f} (P-Valor: {beta_p_val:.4f})\n"
    
    # -------------------------------------------------------------------------
    # RETORNO PARA O CHATBOT
    # -------------------------------------------------------------------------
    
    # Fechar a figura para liberar memória
    plt.close(fig)
    
    return {
        'fig': fig, # O objeto da figura do Matplotlib
        'metricas': df_metricas_formatado, # O DataFrame formatado para o usuário
        'analise_estatistica_md': {'analise': analise_md}, # Dicionário com o markdown
        'analise_estatistica_texto': analise_texto_ia # String para o contexto da IA
    }