from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import yfinance as yf
import numpy as np
import math
import time
import requests
from datetime import datetime, timedelta
import warnings

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"]) 

config = {
    "DEBUG": True,          
    "CACHE_TYPE": "SimpleCache", 
    "CACHE_DEFAULT_TIMEOUT": 21600 # 6 horas em segundos
}
app.config.from_mapping(config)
cache = Cache(app)

# --- CONSTANTES ORIGINAIS ---
FATOR_GRAHAM = 22.5
DY_BAZIN_MIN = 0.13
LISTA_DE_TICKERS_PADRAO = [
    'AMBP3.SA', 'USIM5.SA', 'PETZ3.SA', 'RAIL3.SA', 'HAPV3.SA', 'SIMH3.SA', 'CBAV3.SA', 'CVCB3.SA', 'SUZB3.SA', 'BRAV3.SA',
    'RAIZ4.SA', 'GFSA3.SA', 'MRVE3.SA', 'ONCO3.SA', 'CSNA3.SA', 'BEEF3.SA', 'AZEV4.SA', 'CSAN3.SA', 'BRKM5.SA', 'PCAR3.SA',
    'BHIA3.SA', 'IRBR3.SA', 'AZUL4.SA', 'PRIO3.SA', 'VBBR3.SA', 'JHSF3.SA', 'CMIG4.SA', 'ISAE4.SA', 'COGN3.SA',
    'ECOR3.SA', 'BBAS3.SA', 'BRSR6.SA', 'ABCB4.SA', 'ENGI11.SA', 'GOAU4.SA', 'SBFG3.SA', 'BRAP4.SA', 'VLID3.SA', 'SAPR11.SA', 'CSMG3.SA',
    'POMO4.SA', 'MRFG3.SA', 'CYRE3.SA', 'SMTO3.SA', 'BBDC3.SA', 'GGBR4.SA', 'MYPK3.SA', 'KEPL3.SA', 'CMIN3.SA', 'BBDC4.SA', 'RAPT4.SA',
    'NEOE3.SA', 'ITSA4.SA', 'TAEE11.SA', 'PLPL3.SA', 'VIVA3.SA', 'VALE3.SA', 'VULC3.SA', 'INTB3.SA', 'SBSP3.SA', 'CEAB3.SA', 'EGIE3.SA',
    'SANB11.SA', 'UGPA3.SA', 'ITUB3.SA', 'EZTC3.SA', 'CPFE3.SA', 'MOVI3.SA', 'ALUP11.SA', 'MULT3.SA', 'DIRR3.SA', 'ITUB4.SA', 'ELET3.SA',
    'HYPE3.SA', 'RECV3.SA', 'TTEN3.SA', 'BBSE3.SA', 'PSSA3.SA', 'ELET6.SA', 'CPLE3.SA', 'BRFS3.SA', 'YDUQ3.SA',
    'FLRY3.SA', 'ANIM3.SA', 'LREN3.SA', 'ODPV3.SA', 'UNIP6.SA', 'BPAC11.SA', 'CPLE6.SA', 'PETR4.SA', 'GMAT3.SA', 'CURY3.SA', 'VAMO3.SA',
    'CXSE3.SA', 'PETR3.SA', 'KLBN11.SA', 'MDIA3.SA', 'GGPS3.SA', 'ASAI3.SA', 'B3SA3.SA', 'IGTI11.SA', 'ABEV3.SA', 'MGLU3.SA', 'EQTL3.SA',
    'VIVT3.SA', 'ALOS3.SA', 'AZZA3.SA', 'PORT3.SA', 'STBP3.SA', 'TEND3.SA', 'SLCE3.SA', 'RDOR3.SA', 'SRNA3.SA', 'RENT3.SA',
    'EMBR3.SA', 'DXCO3.SA', 'RADL3.SA', 'SMFT3.SA', 'TOTS3.SA', 'WEGE3.SA', 'AURE3.SA', 'TUPY3.SA', 'LWSA3.SA', 'ALPA4.SA', 'ORVR3.SA',
    'ENEV3.SA'
]

# --- NOVAS CONSTANTES (BACKTEST/RECOMENDAÇÃO) ---
GAIN_THRESHOLD = 0.30
RSI_WINDOW = 14
ALERT_LEVEL = 30
BACKTEST_START_DAYS = 180
STOP_LOSS = -0.05
TAKE_PROFIT = 0.15
MAX_HOLDING_DAYS = 20

# --- FUNÇÕES ORIGINAIS (MANTIDAS) ---

def buscar_dados_fundamentais(tickers):
    dados = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            px = info.get('currentPrice') or info.get('previousClose')
            if not px: continue
            ebitda = info.get('ebitda', 0)
            d_liq = info.get('totalDebt', 0) - info.get('totalCash', 0)
            dados.append({
                "Ticker": ticker, "Nome": info.get('longName', ticker), "Cotacao": px,
                "DY": info.get('dividendYield', 0) or 0, "Payout": info.get('payoutRatio', 0) or 0,
                "DL_EBITDA": d_liq / ebitda if ebitda and ebitda != 0 else 99, "info": info
            })
        except: continue
    return dados

def analisar_bazin(dados):
    df = pd.DataFrame(dados)
    if df.empty: return {"total": 0}, []
    df = df[(df['DY'] >= DY_BAZIN_MIN) & (df['Payout'] <= 0.70) & (df['DL_EBITDA'] <= 2.5)].copy()
    df['Preco_Teto'] = (df['DY'] * df['Cotacao']) / 0.06
    df['Margem'] = (df['Preco_Teto'] / df['Cotacao']) - 1
    df = df.sort_values('Margem', ascending=False)
    return {"total": len(df)}, df[['Ticker', 'Nome', 'Cotacao', 'DY', 'Payout', 'Preco_Teto', 'Margem']].to_dict('records')

def analisar_graham(dados):
    df = pd.DataFrame(dados)
    if df.empty: return {"total": 0}, []
    df['LPA'] = df['info'].apply(lambda x: x.get('trailingEps', 0))
    df['VPA'] = df['info'].apply(lambda x: x.get('bookValue', 0))
    df = df[(df['LPA'] > 0) & (df['VPA'] > 0)].copy()
    df['VI'] = (FATOR_GRAHAM * df['LPA'] * df['VPA']).apply(lambda x: math.sqrt(x) if x > 0 else 0)
    df['Margem'] = (df['VI'] / df['Cotacao']) - 1
    df = df[df['Margem'] > 0].sort_values('Margem', ascending=False)
    return {"total": len(df)}, df[['Ticker', 'Nome', 'Cotacao', 'VI', 'Margem']].to_dict('records')

def analisar_magic_formula(dados):
    res = []
    for d in dados:
        info = d['info']
        ebit = info.get('ebitda', 0) - (info.get('totalCash', 0) * 0.1)
        ev = info.get('enterpriseValue', 0)
        roc = info.get('returnOnAssets', 0)
        if ev and ev > 0 and roc > 0:
            res.append({'Ticker': d['Ticker'], 'Nome': d['Nome'], 'EY': ebit/ev, 'ROC': roc})
    if not res: return {"total": 0}, []
    df = pd.DataFrame(res)
    df['Rank_EY'] = df['EY'].rank(ascending=False)
    df['Rank_ROC'] = df['ROC'].rank(ascending=False)
    df['Score'] = df['Rank_EY'] + df['Rank_ROC']
    return {"total": len(df)}, df.sort_values('Score')[['Ticker', 'Nome', 'Score']].to_dict('records')

def analisar_fiis():
    try:
        url = "https://www.fundamentus.com.br/fii_resultado.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=20)
        df = pd.read_html(r.text, decimal=',', thousands='.')[0]
        df = df.rename(columns={'Papel': 'Ticker'})
        df['DY_Limpo'] = df['Dividend Yield'].str.replace('%', '').str.replace(',', '.').str.strip().astype(float)
        df['Vac_Limpa'] = df['Vacância Média'].str.replace('%', '').str.replace(',', '.').str.strip().astype(float)
        score_p = df['P/VP'].apply(lambda x: 100 if 0.90 <= x <= 1.05 else max(0, 100 - abs(x - 1.00) * 150))
        score_r = df['DY_Limpo'].apply(lambda x: min(100, (x / 8.7) * 100))
        score_v = df['Vac_Limpa'].apply(lambda x: 100 if x <= 10 else max(0, 100 - (x - 10) * 2))
        score_l = df['Liquidez'].apply(lambda x: 100 if x >= 1000000 else (x / 1000000) * 100)
        df['Score'] = round((score_p * 0.3 + score_r * 0.3 + score_v * 0.2 + score_l * 0.2), 2)
        df = df[df['Qtd de imóveis'] >= 5].sort_values('Score', ascending=False)
        return {"status": "sucesso", "total": len(df)}, df[['Ticker', 'Segmento', 'P/VP', 'DY_Limpo', 'Score']].head(15).to_dict('records')
    except Exception as e:
        return {"status": "erro", "detalhe": str(e)}, []

# --- NOVAS FUNÇÕES DE SUPORTE (CALCULO DE RSI E ESTRATÉGIA) ---

def calculate_wilder_rsi(data, window=RSI_WINDOW):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False, min_periods=window).mean()
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100 - (100 / (1 + rs))

def realizar_backtest_e_recomendacao():
    try:
        # 1. Download de Dados
        all_data = yf.download(LISTA_DE_TICKERS_PADRAO, period='2y', interval='1d', progress=False)
        if all_data.empty: return {"erro": "Falha no download de dados"}

        # --- CORREÇÃO DO ERRO DE TIMEZONE ---
        # Remove a informação de fuso horário (torna tz-naive) para permitir comparações
        if all_data.index.tz is not None:
            all_data.index = all_data.index.tz_localize(None)
        # ------------------------------------

        daily_close = all_data['Close']
        daily_high = all_data['High']
        daily_low = all_data['Low']
        
        # 2. Filtragem Rocket Stocks (Universe)
        current_year_start = datetime(datetime.now().year, 1, 1) # Mantido como datetime para comparação
        filtered_tickers = []
        universe_info = []

        for ticker in LISTA_DE_TICKERS_PADRAO:
            try:
                df_t = daily_close[ticker].dropna()
                if len(df_t) < 120: continue
                
                curr_px = df_t.iloc[-1]
                gain_3m = (curr_px - df_t.iloc[-60]) / df_t.iloc[-60] if len(df_t) > 60 else 0
                gain_6m = (curr_px - df_t.iloc[-120]) / df_t.iloc[-120] if len(df_t) > 120 else 0
                
                # Comparação agora funciona pois ambos são tz-naive
                ytd_data = df_t[df_t.index >= current_year_start]
                gain_ytd = (curr_px - ytd_data.iloc[0]) / ytd_data.iloc[0] if not ytd_data.empty else 0
                
                rockets = sum([gain_3m >= GAIN_THRESHOLD, gain_6m >= GAIN_THRESHOLD, gain_ytd >= GAIN_THRESHOLD])
                
                if rockets > 0:
                    filtered_tickers.append(ticker)
                    universe_info.append({
                        "Ticker": ticker, 
                        "Rockets": int(rockets), 
                        "3m": round(float(gain_3m), 4), 
                        "6m": round(float(gain_6m), 4), 
                        "YTD": round(float(gain_ytd), 4)
                    })
            except:
                continue

        # 3. Simulação de Backtest
        open_trades = {}
        closed_trades = []
        # Pega os últimos dias disponíveis
        backtest_dates = daily_close.index[-(BACKTEST_START_DAYS):].tolist()

        for current_date in backtest_dates:
            # Fechamento de trades existentes
            for ticker in list(open_trades.keys()):
                trade = open_trades[ticker]
                px_close = daily_close.loc[current_date, ticker]
                px_high = daily_high.loc[current_date, ticker]
                px_low = daily_low.loc[current_date, ticker]
                
                sl_px = trade['Entry Price'] * (1 + STOP_LOSS)
                tp_px = trade['Entry Price'] * (1 + TAKE_PROFIT)
                exit_reason, exit_px = None, None

                if px_low <= sl_px: 
                    exit_reason, exit_px = 'Stop Loss', sl_px
                elif px_high >= tp_px: 
                    exit_reason, exit_px = 'Take Profit', tp_px
                
                trade['Days Held'] += 1
                if not exit_reason and trade['Days Held'] >= MAX_HOLDING_DAYS:
                    exit_reason, exit_px = 'Time Out', px_close

                if exit_reason:
                    trade.update({
                        'Exit Date': current_date.strftime('%Y-%m-%d'), 
                        'Exit Price': round(float(exit_px), 2), 
                        'Return': round(float((exit_px / trade['Entry Price']) - 1), 4), 
                        'Exit Reason': exit_reason
                    })
                    closed_trades.append(trade)
                    del open_trades[ticker]

            # Abertura de novos trades (Sinais)
            hist_data = daily_close.loc[:current_date]
            for ticker in filtered_tickers:
                if ticker in open_trades: continue
                
                series_t = hist_data[ticker].dropna()
                if len(series_t) < RSI_WINDOW: continue
                
                rsi_series = calculate_wilder_rsi(series_t.to_frame(name='Close'))
                if rsi_series.empty: continue
                
                rsi_val = rsi_series.iloc[-1]
                if rsi_val <= ALERT_LEVEL:
                    open_trades[ticker] = {
                        'Ticker': ticker, 
                        'Entry Date': current_date.strftime('%Y-%m-%d'),
                        'Entry Price': round(float(series_t.iloc[-1]), 2), 
                        'Entry RSI': round(float(rsi_val), 2), 
                        'Days Held': 0
                    }

        # 4. Alerta Hoje
        alertas = []
        if not daily_close.empty:
            for ticker in filtered_tickers:
                series = daily_close[ticker].dropna()
                if len(series) < RSI_WINDOW: continue
                rsi_now = calculate_wilder_rsi(series.to_frame(name='Close')).iloc[-1]
                if rsi_now <= ALERT_LEVEL:
                    alertas.append({
                        "Ticker": ticker, 
                        "RSI": round(float(rsi_now), 2), 
                        "Preco": round(float(series.iloc[-1]), 2)
                    })

        # Resultados do Backtest
        df_results = pd.DataFrame(closed_trades)
        resumo = {"win_rate": 0, "retorno_acumulado": 0, "total_trades": 0}
        
        if not df_results.empty:
            win_rate = len(df_results[df_results['Return'] > 0]) / len(df_results)
            # Simulação de retorno composto simples
            cum_ret = (1 + df_results['Return']).prod() - 1
            resumo = {
                "win_rate": round(float(win_rate * 100), 2), 
                "retorno_acumulado": round(float(cum_ret * 100), 2), 
                "total_trades": len(df_results)
            }

        return {
            "universe": universe_info,
            "resumo_backtest": resumo,
            "detalhes_trades": df_results.to_dict('records') if not df_results.empty else [],
            "alertas_hoje": alertas
        }
    except Exception as e:
        return {"erro": f"Erro interno: {str(e)}"}

# --- ROTAS DA API ---

@app.route('/analise-completa', methods=['GET'])
@cache.cached(timeout=21600, query_string=True)
def api_unificada():
    t_param = request.args.get('ticker')
    tickers = [t_param] if t_param else LISTA_DE_TICKERS_PADRAO

    print("--- PROCESSANDO DADOS (Sem cache) ---")
    
    # Executa lógica original
    dados_acoes = buscar_dados_fundamentais(tickers)
    m_bazin, d_bazin = analisar_bazin(dados_acoes)
    m_graham, d_graham = analisar_graham(dados_acoes)
    m_magic, d_magic = analisar_magic_formula(dados_acoes)
    m_fiis, d_fiis = analisar_fiis()

    # Executa nova lógica de Backtest e Recomendação
    recomendacoes = realizar_backtest_e_recomendacao()

    return jsonify({
        "data": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bazin": {"resumo": m_bazin, "dados": d_bazin},
        "graham": {"resumo": m_graham, "dados": d_graham},
        "magic_formula": {"resumo": m_magic, "dados": d_magic},
        "fiis": {"resumo": m_fiis, "dados": d_fiis},
        "estrategia_swing_trade": recomendacoes
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)