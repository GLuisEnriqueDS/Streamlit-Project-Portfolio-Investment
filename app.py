# basic pyhton librarioes
import streamlit as st
import pandas as pd
import numpy as np
from datetime import  date

# libraries to retrieve and download data
import requests
from io import BytesIO
import yfinance  as yf
import base64
import copy

# ploting libraries
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache_resource
def get_session():
    return SessionState(
        df=pd.DataFrame(), 
        data=pd.DataFrame(), 
        portfolio=pd.DataFrame(), 
        backtest=pd.DataFrame(), 
        optimized_data=pd.DataFrame()
    )

session_state = get_session()

def download_data(data, period):
    dfs = []
    if isinstance(data, dict):
        for name, ticker in data.items():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{name}_{col}" for col in hist.columns]  # Add prefix to the name
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist)
    elif isinstance(data, list):
        for ticker in data:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{ticker}_{col}" for col in hist.columns]  # Add prefix to the name
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist)

    combined_df = pd.concat(dfs, axis=1, join='outer')  # Use join='outer' to handle different time indices
    return combined_df

def fill_moving_avg(df, window_size, method='gap'):
    if method == 'gap':
        date_index = df.index
        df.reset_index(drop=True, inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            nan_indices = df[df[col].isna()].index
            for index in nan_indices:
                start = max(0, index - window_size)
                end = index + 1
                window_data = df[col].iloc[start:end]
                mean_value = round(window_data.mean(), 4)
                df.at[index, col] = mean_value
        df.index = date_index
    else:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[col] = df[col].fillna(method='bfill')
    return df

def corr_and_cov(df):
    corr_df = df.corr().round(2)
    st.write("Correlation Between Assets:")
    st.write(corr_df)
    cov_df = df.cov().round(2)
    st.write("Covariance Between Assets:")
    st.write(cov_df)

def plot_cum_returns(df, weights):    
    daily_cum_returns = (1 + df.dropna().pct_change()).cumprod() * 100
    portfolio_returns = (df.dot(weights)).pct_change()
    portfolio_cum_returns = (1 + portfolio_returns).cumprod() * 100
    daily_cum_returns['Portfolio'] = portfolio_cum_returns
    fig = px.line(daily_cum_returns, title='Cumulative Returns of Individual Stocks and Portfolio')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig)

def EfficientFrontier(data):
    df = data.copy()
    assets = df.columns
    port_returns = []
    port_volatility = []
    stock_weights = []
    sharpe_ratio = []

    returns_daily = df.pct_change()
    log_returns = np.log(returns_daily + 1)
    returns_annual = np.exp(log_returns.mean() * 252) - 1
    ann_risk = returns_daily.std() * np.sqrt(252)

    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    num_assets = len(data.columns)
    num_portfolios = 1000


    np.random.seed(101)

    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                'Volatility': port_volatility,
                'Sharpe Ratio': sharpe_ratio}

    for counter,symbol in enumerate(assets):
        portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

    df = pd.DataFrame(portfolio)

    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in assets]

    df = df[column_order]
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_return = df['Returns'].max()

    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    max_return_port = df.loc[df['Returns'] == max_return]

    optimal_weights = sharpe_portfolio.filter(like='Weight').values.flatten()

 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Volatility'],
        y=df['Returns'],
        mode='markers',
        marker=dict(
            color=df['Sharpe Ratio'],  
            colorscale='Viridis',  
            size=5,
            line=dict(width=1, color='black')  
        ),
        name='Portafolios'
    ))

    fig.add_trace(go.Scatter(
        x=[sharpe_portfolio['Volatility'].values[0]],  # extrae el valor escalar
        y=[sharpe_portfolio['Returns'].values[0]],     # extrae el valor escalar
        mode='markers',
        marker=dict(
            color='red',
            size=10,  # aumenta el tamaño para mejor visualización
            symbol='diamond'
        ),
        name='Máx Sharpe Ratio'
    ))

    fig.add_trace(go.Scatter(
        x=[min_variance_port['Volatility'].values[0]],  # extrae el valor escalar
        y=[min_variance_port['Returns'].values[0]],     # extrae el valor escalar
        mode='markers',
        marker=dict(
            color='blue',
            size=10,  
            symbol='diamond'
        ),
        name='Mín Volatility'
    ))

    fig.add_trace(go.Scatter(
        x=[max_return_port['Volatility'].values[0]],  # extrae el valor escalar
        y=[max_return_port['Returns'].values[0]],     # extrae el valor escalar
        mode='markers',
        marker=dict(
            color='yellow',
            size=10,  
            symbol='diamond'
        ),
        name='Máx Return'
    ))

    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Std. Deviation)',
        yaxis_title='Expected Returns',
        showlegend=True,
        width=800,
        height=600
    )


    st.plotly_chart(fig)

    min_variance_port = min_variance_port.T
    sharpe_portfolio = sharpe_portfolio.T
    max_return_port = max_return_port.T

    min_variance_port.columns = ['Portfolio Min Volatility']
    sharpe_portfolio.columns = ['Portfolio Max Sharpe Ratio']
    max_return_port.columns = ['Portfolio Max Return']

    combined_df = pd.concat([min_variance_port, sharpe_portfolio, max_return_port], axis=1)
    st.write(combined_df)

    return df, optimal_weights

st.title("Investment Portfolio Optimization")

selected_timeframes = st.selectbox('Select Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=7)

currencies_dict  =  {'USD/JPY': 'USDJPY=X', 'USD/BRL': 'BRL=X', 'USD/ARS': 'ARS=X', 'USD/PYG': 'PYG=X', 'USD/UYU': 'UYU=X',
                     'USD/CNY': 'CNY=X', 'USD/KRW': 'KRW=X', 'USD/MXN': 'MXN=X', 'USD/IDR': 'IDR=X', 'USD/EUR': 'EUR=X',
                     'USD/CAD': 'CAD=X', 'USD/GBP': 'GBP=X', 'USD/CHF': 'CHF=X', 'USD/AUD': 'AUD=X', 'USD/NZD': 'NZD=X',
                     'USD/HKD': 'HKD=X', 'USD/SGD': 'SGD=X', 'USD/INR': 'INR=X', 'USD/RUB': 'RUB=X', 'USD/ZAR': 'ZAR=X',
                     'USD/SEK': 'SEK=X', 'USD/NOK': 'NOK=X', 'USD/TRY': 'TRY=X', 'USD/AED': 'AED=X', 'USD/SAR': 'SAR=X',
                     'USD/THB': 'THB=X', 'USD/DKK': 'DKK=X', 'USD/MYR': 'MYR=X', 'USD/PLN': 'PLN=X', 'USD/EGP': 'EGP=X',
                     'USD/CZK': 'CZK=X', 'USD/ILS': 'ILS=X', 'USD/HUF': 'HUF=X', 'USD/PHP': 'PHP=X', 'USD/CLP': 'CLP=X',
                     'USD/COP': 'COP=X', 'USD/PEN': 'PEN=X', 'USD/KWD': 'KWD=X', 'USD/QAR': 'USD/QAR'
                    }
crypto_dict = {'Bitcoin USD': 'BTC-USD', 'Ethereum USD': 'ETH-USD', 'Tether USDT USD': 'USDT-USD',
               'Bnb USD': 'BNB-USD', 'Solana USD': 'SOL-USD', 'Xrp USD': 'XRP-USD', 'Usd Coin USD': 'USDC-USD',
               'Lido Staked Eth USD': 'STETH-USD', 'Cardano USD': 'ADA-USD', 'Avalanche USD': 'AVAX-USD',
               'Dogecoin USD': 'DOGE-USD', 'Wrapped Tron USD': 'WTRX-USD', 'Tron USD': 'TRX-USD',
               'Polkadot USD': 'DOT-USD', 'Chainlink USD': 'LINK-USD', 'Toncoin USD': 'TON11419-USD',
               'Polygon USD': 'MATIC-USD', 'Wrapped Bitcoin USD': 'WBTC-USD', 'Shiba Inu USD': 'SHIB-USD',
               'Internet Computer USD': 'ICP-USD', 'Dai USD': 'DAI-USD', 'Litecoin USD': 'LTC-USD',
               'Bitcoin Cash USD': 'BCH-USD', 'Uniswap USD': 'UNI7083-USD', 'Cosmos USD': 'ATOM-USD',
               'Unus Sed Leo USD': 'LEO-USD', 'Ethereum Classic USD': 'ETC-USD', 'Stellar USD': 'XLM-USD',
               'Okb USD': 'OKB-USD', 'Near Protocol USD': 'NEAR-USD', 'Optimism USD': 'OP-USD',
               'Injective USD': 'INJ-USD', 'Aptos USD': 'APT21794-USD', 'Monero USD': 'XMR-USD',
               'Filecoin USD': 'FIL-USD', 'Lido Dao USD': 'LDO-USD', 'Celestia USD': 'TIA22861-USD',
               'Hedera USD': 'HBAR-USD', 'Wrapped Hbar USD': 'WHBAR-USD', 'Immutable USD': 'IMX10603-USD',
               'Wrapped Eos USD': 'WEOS-USD', 'Arbitrum USD': 'ARB11841-USD', 'Kaspa USD': 'KAS-USD',
               'Bitcoin Bep2 USD': 'BTCB-USD', 'Stacks USD': 'STX4847-USD', 'Mantle USD': 'MNT27075-USD',
               'First Digital Usd Usd': 'FDUSD-USD', 'Vechain USD': 'VET-USD', 'Cronos USD': 'CRO-USD',
               'Wrapped Beacon Eth USD': 'WBETH-USD', 'Trueusd USD': 'TUSD-USD', 'Sei USD': 'SEI-USD',
               'Maker USD': 'MKR-USD', 'Hex USD': 'HEX-USD', 'Rocket Pool Eth USD': 'RETH-USD',
               'Bitcoin Sv USD': 'BSV-USD', 'Render USD': 'RNDR-USD', 'Bittensor USD': 'TAO22974-USD',
               'The Graph USD': 'GRT6719-USD', 'Algorand USD': 'ALGO-USD', 'Ordi USD': 'ORDI-USD',
               'Aave USD': 'AAVE-USD', 'Thorchain USD': 'RUNE-USD', 'Quant USD': 'QNT-USD',
               'Multiversx USD': 'EGLD-USD', 'Sui USD': 'SUI20947-USD', 'Mina USD': 'MINA-USD',
               'Sats USD': '1000SATS-USD', 'Flow USD': 'FLOW-USD', 'Helium USD': 'HNT-USD',
               'Fantom USD': 'FTM-USD', 'Synthetix USD': 'SNX-USD', 'The Sandbox USD': 'SAND-USD',
               'Theta Network USD': 'THETA-USD', 'Axie Infinity USD': 'AXS-USD', 'Tezos USD': 'XTZ-USD',
               'Beam USD': 'BEAM28298-USD', 'Bittorrent(New) USD': 'BTT-USD', 'Kucoin Token USD': 'KCS-USD',
               'Dydx (Ethdydx) USD': 'ETHDYDX-USD', 'Ftx Token USD': 'FTT-USD', 'Astar USD': 'ASTR-USD',
               'Wemix USD': 'WEMIX-USD', 'Blur USD': 'BLUR-USD', 'Cheelee USD': 'CHEEL-USD',
               'Chiliz USD': 'CHZ-USD', 'Bitget Token USD': 'BGB-USD', 'Decentraland USD': 'MANA-USD',
               'Neo USD': 'NEO-USD', 'Osmosis USD': 'OSMO-USD', 'Eos USD': 'EOS-USD', 'Bonk USD': 'BONK-USD',
               'Kava USD': 'KAVA-USD', 'Woo USD': 'WOO-USD', 'Klaytn USD': 'KLAY-USD', 'Flare USD': 'FLR-USD',
               'Oasis Network USD': 'ROSE-USD', 'Iota USD': 'IOTA-USD', 'Usdd USD': 'USDD-USD',
               'Terra Classic USD': 'LUNC-USD'}
commodities_dict = { "BRENT CRUDE OIL LAST DAY FINANC": "BZ=F", "COCOA": "CC=F", "COFFEE": "KC=F", "COPPER": "HG=F",
                    "CORN FUTURES": "ZC=F", "COTTON": "CT=F", "HEATING OIL": "HO=F", "KC HRW WHEAT FUTURES": "KE=F",
                    "LEAN HOGS FUTURES": "HE=F", "LIVE CATTLE FUTURES": "LE=F", "MONT BELVIEU LDH PROPANE (OPIS)": "B0=F",
                    "NATURAL GAS": "NG=F", "ORANGE JUICE": "OJ=F", "GOLD": "GC=F", "OAT FUTURES": "ZO=F",
                    "PALLADIUM": "PA=F", "CRUDE OIL": "CL=F", "PLATINUM": "PL=F", "RBOB GASOLINE": "RB=F",
                    "RANDOM LENGTH LUMBER FUTURES": "LBS=F", "ROUGH RICE FUTURES": "ZR=F", "SILVER": "SI=F",
                    "SOYBEAN FUTURES": "ZS=F", "SOYBEAN OIL FUTURES": "ZL=F", "S&P COMPOSITE 1500 ESG TILTED I": "ZM=F",
                    "SUGAR": "SB=F", "WISDOMTREE INTERNATIONAL HIGH D": "GF=F"
                }
b3_stocks = {"3m": "MMMC34.SA", "Aes brasil": "AESB3.SA", "Af invest": "AFHI11.SA", "Afluente t": "AFLT3.SA", "Agribrasil": "GRAO3.SA",
    "Agogalaxy": "AGXY3.SA", "Alliar": "AALR3.SA", "Alper": "APER3.SA", "Google": "GOGL35.SA", "Alupar investimento": "ALUP4.SA",
    "American express": "AXPB34.SA", "Arcelor": "ARMT34.SA", "Att inc": "ATTB34.SA", "Auren energia": "AURE3.SA", "Banco do brasil": "BBAS3.SA",
    "Banco mercantil de investimentos": "BMIN3.SA", "Banco pan": "BPAN4.SA", "Bank america": "BOAC34.SA", "Banrisul": "BRSR3.SA",
    "Baumer": "BALM3.SA", "Bb seguridade": "BBSE3.SA", "Biomm": "BIOM3.SA", "Bmg": "BMGB4.SA", "Caixa agências": "CXAG11.SA",
    "Camden prop": "C2PT34.SA", "Camil": "CAML3.SA", "Carrefour": "CRFB3.SA", "Cartesia fiici": "CACR11.SA", "Casan": "CASN4.SA",
    "Ceb": "CEBR6.SA", "Ceee-d": "CEED4.SA", "Ceg": "CEGR3.SA", "Celesc": "CLSC4.SA", "Cemig": "CMIG4.SA", "Chevron": "CHVX34.SA",
    "Churchill dw": "C2HD34.SA", "Cisco": "CSCO34.SA", "Citigroup": "CTGP34.SA", "Clearsale": "CLSA3.SA", "Coca-cola": "COCA34.SA",
    "Coelce": "COCE6.SA", "Coinbase glob": "C2OI34.SA", "Colgate": "COLG34.SA", "Comgás": "CGAS3.SA", "Conocophillips": "COPH34.SA",
    "Copel": "CPLE6.SA", "Cpfl energia": "CPFE3.SA", "Csn": "CSNA3.SA", "Dexco": "DXCO3.SA", "Dexxos part": "DEXP3.SA",
    "Dimed": "PNVL3.SA", "Doordash inc": "D2AS34.SA", "Draftkings": "D2KN34.SA", "Ebay": "EBAY34.SA", "Enauta part": "ENAT3.SA",
    "Energisa mt": "ENMT3.SA", "Engie brasil": "EGIE3.SA", "Eqi receci": "EQIR11.SA", "Eucatex": "EUCA4.SA", "Exxon mobil": "EXXO34.SA",
    "Ferbasa": "FESA4.SA", "Fiagro jgp ci": "JGPX11.SA", "Fiagro riza ci": "RZAG11.SA", "Fii brio me ci": "BIME11.SA", "Fii cyrela ci es": "CYCR11.SA",
    "Fii gtis lg": "GTLG11.SA", "Fii husi ci es": "HUSI11.SA", "Fii js a finci": "JSAF11.SA", "Fii more crici er": "MORC11.SA", "Fii rooftopici": "ROOF11.SA",
    "Fleury": "FLRY3.SA", "Freeport": "FCXO34.SA", "Ft cloud cpt": "BKYY39.SA", "Ft dj intern": "BFDN39.SA", "Ft nasd cyber": "BCIR39.SA",
    "Ft nasd100 eq": "BQQW39.SA", "Ft nasd100 tc": "BQTC39.SA", "Ft nat gas": "BFCG39.SA", "Ft nyse biot drn": "BFBI39.SA", "Ft risi divid": "BFDA39.SA",
    "G2d investments": "G2DI33.SA", "Ge": "GEOO34.SA", "General shopping": "GSHP3.SA", "Gerd paranapanema": "GEPA4.SA", "Golias": "GOAU4.SA",
    "Godaddy inc": "G2DD34.SA", "Goldman sachs": "GSGI34.SA", "Grd": "IGBR3.SA", "Halliburton": "HALI34.SA", "Honeywell": "HONB34.SA",
    "Hp company": "HPQB34.SA", "Hypera pharma": "HYPE3.SA", "Ibm": "IBMB34.SA", "Iguatemi s.a.": "IGTI3.SA", "Infracommerce": "IFCM3.SA",
    "Intel": "ITLC34.SA", "Investo alug": "ALUG11.SA", "Investo ustk": "USTK11.SA", "Investo wrld": "WRLD11.SA", "Irb brasil re": "IRBR3.SA",
    "Isa cteep": "TRPL4.SA", "Itaú unibanco": "ITUB4.SA", "Itaúsa": "ITSA4.SA", "Jbs": "JBSS3.SA", "Johnson": "JNJB34.SA",
    "Jpmorgan": "JPMC34.SA", "Kingsoft chl": "K2CG34.SA", "Klabin s/a": "KLBN11.SA", "Livetech": "LVTC3.SA", "Locaweb": "LWSA3.SA",
    "Log": "LOGG3.SA", "Lps brasil": "LPSB3.SA", "Marfrig": "MRFG3.SA", "Mastercard": "MSCD34.SA", "Mdiasbranco": "MDIA3.SA",
    "Melnick": "MELK3.SA", "Meliuz": "CASH3.SA", "Mercado livre": "MELI34.SA", "Microsoft": "MSFT34.SA", "Mrv engenharia": "MRVE3.SA",
    "Natura": "NTCO3.SA", "Netflix": "NFLX34.SA", "Oi": "OIBR3.SA", "Oracle": "ORCL34.SA", "Pão de açúcar": "PCAR3.SA",
    "Petrobras": "PETR4.SA", "Petróleo": "PEAB3.SA", "Pfizer": "PFIZ34.SA", "Plascar": "PLAS3.SA", "Porto seguro": "PSSA3.SA",
    "Positivo": "POSI3.SA", "Procter": "PGCO34.SA", "Qualicorp": "QUAL3.SA", "Randon": "RAPT4.SA", "Raia drogasil": "RADL3.SA",
    "Renner": "LREN3.SA", "Rossi": "RSID3.SA", "Rumo s.a.": "RAIL3.SA", "Santander": "SANB11.SA", "Telefônica": "VIVT3.SA",
    "Tim": "TIMS3.SA", "Totvs": "TOTS3.SA", "Trisul": "TRIS3.SA", "Ultrapar": "UGPA3.SA", "Unipar": "UNIP6.SA", "Usiminas": "USIM5.SA",
    "Vale": "VALE3.SA", "Vivara": "VIVA3.SA", "Vulcabras": "VULC3.SA", "Weg": "WEGE3.SA", "Whirlpool": "WHRL3.SA", "Yduqs": "YDUQ3.SA"
}
indexes_dict ={'S&P GSCI': 'GD=F', 'IBOVESPA': '^BVSP', 'S&P/CLX IPSA': '^IPSA',
                    'MERVAL': '^MERV', 'IPC MEXICO': '^MXX', 'S&P 500': '^GSPC',
                    'Dow Jones Industrial Average': '^DJI', 'NASDAQ Composite': '^IXIC',
                    'NYSE COMPOSITE (DJ)': '^NYA', 'NYSE AMEX COMPOSITE INDEX': '^XAX',
                    'Russell 2000': '^RUT', 'CBOE Volatility Index': '^VIX',
                    'S&P/TSX Composite index': '^GSPTSE', 'FTSE 100': '^FTSE',
                    'DAX PERFORMANCE-INDEX': '^GDAXI', 'CAC 40': '^FCHI',
                    'ESTX 50 PR.EUR': '^STOXX50E', 'Euronext 100 Index': '^N100',
                    'BEL 20': '^BFX', 'MOEX Russia Index': 'IMOEX.ME', 'Nikkei 225': '^N225',
                    'HANG SENG INDEX': '^HSI', 'SSE Composite Index': '000001.SS',
                    'Shenzhen Index': '399001.SZ', 'STI Index': '^STI', 'S&P/ASX 200': '^AXJO',
                    'ALL ORDINARIES': '^AORD', 'S&P BSE SENSEX': '^BSESN', 'IDX COMPOSITE': '^JKSE',
                    'FTSE Bursa Malaysia KLCI': '^KLSE', 'S&P/NZX 50 INDEX GROSS': '^NZ50',
                    'KOSPI Composite Index': '^KS11', 'TSEC weighted index': '^TWII',
                    'TA-125': '^TA125.TA', 'Top 40 USD Net TRI Index': '^JN0U.JO', 'NIFTY 50': '^NSEI'
                    }
sp500_dict = {'3M': 'MMM', 'A. O. Smith': 'AOS', 'Abbott': 'ABT', 'AbbVie': 'ABBV', 'Accenture': 'ACN', 'Adobe Inc.': 'ADBE',
              'Advanced Micro Devices': 'AMD', 'AES Corporation': 'AES', 'Aflac': 'AFL', 'Agilent Technologies': 'A', 'Air Products and Chemicals': 'APD',
              'Airbnb': 'ABNB', 'Akamai': 'AKAM', 'Albemarle Corporation': 'ALB', 'Alexandria Real Estate Equities': 'ARE', 'Align Technology': 'ALGN',
              'Allegion': 'ALLE', 'Alliant Energy': 'LNT', 'Allstate': 'ALL', 'Google': 'GOOGL', 'Google': 'GOOG',
              'Altria': 'MO', 'Amazon': 'AMZN', 'Amcor': 'AMCR', 'Ameren': 'AEE', 'American Airlines Group': 'AAL', 'American Electric Power': 'AEP',
              'American Express': 'AXP', 'American International Group': 'AIG', 'American Tower': 'AMT', 'American Water Works': 'AWK', 'Ameriprise Financial': 'AMP',
              'AMETEK': 'AME', 'Amgen': 'AMGN', 'Amphenol': 'APH', 'Analog Devices': 'ADI', 'ANSYS': 'ANSS', 'Aon': 'AON',
              'APA Corporation': 'APA', 'Apple Inc.': 'AAPL', 'Applied Materials': 'AMAT', 'Aptiv': 'APTV', 'Arch Capital Group': 'ACGL', 'Archer-Daniels-Midland': 'ADM',
              'Arista Networks': 'ANET', 'Arthur J. Gallagher & Co.': 'AJG', 'Assurant': 'AIZ', 'AT&T': 'T', 'Atmos Energy': 'ATO', 'Autodesk': 'ADSK',
              'Automated Data Processing': 'ADP', 'AutoZone': 'AZO', 'Avalonbay Communities': 'AVB', 'Avery Dennison': 'AVY', 'Axon Enterprise': 'AXON', 'Baker Hughes': 'BKR',
              'Ball Corporation': 'BALL', 'Bank of America': 'BAC', 'Bank of New York Mellon': 'BK', 'Bath & Body Works, Inc.': 'BBWI', 'Baxter International': 'BAX', 'Becton Dickinson': 'BDX',
              'Berkshire Hathaway': 'BRK.B', 'Best Buy': 'BBY', 'Bio-Rad': 'BIO', 'Bio-Techne': 'TECH', 'Biogen': 'BIIB', 'BlackRock': 'BLK', 'Blackstone': 'BX',
              'Boeing': 'BA', 'Booking Holdings': 'BKNG', 'BorgWarner': 'BWA', 'Boston Properties': 'BXP', 'Boston Scientific': 'BSX', 'Bristol Myers Squibb': 'BMY', 'Broadcom Inc.': 'AVGO',
              'Broadridge Financial Solutions': 'BR', 'Brown & Brown': 'BRO', 'Brown–Forman': 'BF.B', 'Builders FirstSource': 'BLDR', 'Bunge Global SA': 'BG', 'Cadence Design Systems': 'CDNS',
              'Caesars Entertainment': 'CZR', 'Camden Property Trust': 'CPT', 'Campbell Soup Company': 'CPB', 'Capital One': 'COF', 'Cardinal Health': 'CAH', 'CarMax': 'KMX',
              'Carnival': 'CCL', 'Carrier Global': 'CARR', 'Catalent': 'CTLT', 'Caterpillar Inc.': 'CAT', 'Cboe Global Markets': 'CBOE', 'CBRE Group': 'CBRE', 'CDW': 'CDW',
              'Celanese': 'CE', 'Cencora': 'COR', 'Centene Corporation': 'CNC', 'CenterPoint Energy': 'CNP', 'Ceridian': 'CDAY', 'CF Industries': 'CF', 'CH Robinson': 'CHRW',
              'Charles River Laboratories': 'CRL', 'Charles Schwab Corporation': 'SCHW', 'Charter Communications': 'CHTR', 'Chevron Corporation': 'CVX', 'Chipotle Mexican Grill': 'CMG',
              'Chubb Limited': 'CB', 'Church & Dwight': 'CHD', 'Cigna': 'CI', 'Cincinnati Financial': 'CINF', 'Cintas': 'CTAS', 'Cisco': 'CSCO', 'Citigroup': 'C',
              'Citizens Financial Group': 'CFG', 'Clorox': 'CLX', 'CME Group': 'CME', 'CMS Energy': 'CMS', 'Coca-Cola Company (The)': 'KO', 'Cognizant': 'CTSH', 'Colgate-Palmolive': 'CL',
              'Comcast': 'CMCSA', 'Comerica': 'CMA', 'Conagra Brands': 'CAG', 'ConocoPhillips': 'COP', 'Consolidated Edison': 'ED', 'Constellation Brands': 'STZ', 'Constellation Energy': 'CEG',
              'CooperCompanies': 'COO', 'Copart': 'CPRT', 'Corning Inc.': 'GLW', 'Corteva': 'CTVA', 'CoStar Group': 'CSGP', 'Costco': 'COST', 'Coterra': 'CTRA', 'Crown Castle': 'CCI',
              'CSX': 'CSX', 'Cummins': 'CMI', 'CVS Health': 'CVS', 'Danaher Corporation': 'DHR', 'Darden Restaurants': 'DRI', 'DaVita Inc.': 'DVA', 'John Deere': 'DE', 'Delta Air Lines': 'DAL',
              'Dentsply Sirona': 'XRAY', 'Devon Energy': 'DVN', 'Dexcom': 'DXCM', 'Diamondback Energy': 'FANG', 'Digital Realty': 'DLR', 'Discover Financial': 'DFS', 'Dollar General': 'DG',
              'Dollar Tree': 'DLTR', 'Dominion Energy': 'D', 'Domino\'s': 'DPZ', 'Dover Corporation': 'DOV', 'Dow Inc.': 'DOW', 'DR Horton': 'DHI', 'DTE Energy': 'DTE', 'Duke Energy': 'DUK',
              'Dupont': 'DD', 'Eastman Chemical Company': 'EMN', 'Eaton Corporation': 'ETN', 'eBay': 'EBAY', 'Ecolab': 'ECL', 'Edison International': 'EIX', 'Edwards Lifesciences': 'EW',
              'Electronic Arts': 'EA', 'Elevance Health': 'ELV', 'Eli Lilly and Company': 'LLY', 'Emerson Electric': 'EMR', 'Enphase': 'ENPH', 'Entergy': 'ETR', 'EOG Resources': 'EOG',
              'EPAM Systems': 'EPAM', 'EQT': 'EQT', 'Equifax': 'EFX', 'Equinix': 'EQIX', 'Equity Residential': 'EQR', 'Essex Property Trust': 'ESS', 'Estée Lauder Companies (The)': 'EL',
              'Etsy': 'ETSY', 'Everest Re': 'EG', 'Evergy': 'EVRG', 'Eversource': 'ES', 'Exelon': 'EXC', 'Expedia Group': 'EXPE', 'Expeditors International': 'EXPD', 'Extra Space Storage': 'EXR',
              'ExxonMobil': 'XOM', 'F5, Inc.': 'FFIV', 'FactSet': 'FDS', 'Fair Isaac': 'FICO', 'Fastenal': 'FAST', 'Federal Realty': 'FRT', 'FedEx': 'FDX', 'Fidelity National Information Services': 'FIS',
              'Fifth Third Bank': 'FITB', 'First Solar': 'FSLR', 'FirstEnergy': 'FE', 'Fiserv': 'FI', 'FleetCor': 'FLT', 'FMC Corporation': 'FMC', 'Ford Motor Company': 'F', 'Fortinet': 'FTNT',
              'Fortive': 'FTV', 'Fox Corporation (Class A)': 'FOXA', 'Fox Corporation (Class B)': 'FOX', 'Franklin Templeton': 'BEN', 'Freeport-McMoRan': 'FCX', 'Garmin': 'GRMN', 'Gartner': 'IT',
              'GE Healthcare': 'GEHC', 'Gen Digital': 'GEN', 'Generac': 'GNRC', 'General Dynamics': 'GD', 'General Electric': 'GE', 'General Mills': 'GIS', 'General Motors': 'GM', 'Genuine Parts Company': 'GPC',
              'Gilead Sciences': 'GILD', 'Global Payments': 'GPN', 'Globe Life': 'GL', 'Goldman Sachs': 'GS', 'Halliburton': 'HAL', 'Hartford (The)': 'HIG', 'Hasbro': 'HAS', 'HCA Healthcare': 'HCA',
              'Healthpeak': 'PEAK', 'Henry Schein': 'HSIC', 'Hershey\'s': 'HSY', 'Hess Corporation': 'HES', 'Hewlett Packard Enterprise': 'HPE', 'Hilton Worldwide': 'HLT', 'Hologic': 'HOLX',
              'Home Depot (The)': 'HD', 'Honeywell': 'HON', 'Hormel Foods': 'HRL', 'Host Hotels & Resorts': 'HST', 'Howmet Aerospace': 'HWM', 'HP Inc.': 'HPQ', 'Hubbell Incorporated': 'HUBB',
              'Humana': 'HUM', 'Huntington Bancshares': 'HBAN', 'Huntington Ingalls Industries': 'HII', 'IBM': 'IBM', 'IDEX Corporation': 'IEX', 'IDEXX Laboratories': 'IDXX',
              'Illinois Tool Works': 'ITW', 'Illumina': 'ILMN', 'Incyte': 'INCY', 'Ingersoll Rand': 'IR', 'Insulet': 'PODD', 'Intel': 'INTC', 'Intercontinental Exchange': 'ICE',
              'International Flavors & Fragrances': 'IFF', 'International Paper': 'IP', 'Interpublic Group of Companies (The)': 'IPG', 'Intuit': 'INTU', 'Intuitive Surgical': 'ISRG',
              'Invesco': 'IVZ', 'Invitation Homes': 'INVH', 'IQVIA': 'IQV', 'Iron Mountain': 'IRM', 'J.B. Hunt': 'JBHT', 'Jabil': 'JBL', 'Jack Henry & Associates': 'JKHY', 'Jacobs Solutions': 'J',
              'Johnson & Johnson': 'JNJ', 'Johnson Controls': 'JCI', 'JPMorgan Chase': 'JPM', 'Juniper Networks': 'JNPR', 'Kellanova': 'K', 'Kenvue': 'KVUE', 'Keurig Dr Pepper': 'KDP',
              'KeyCorp': 'KEY', 'Keysight': 'KEYS', 'Kimberly-Clark': 'KMB', 'Kimco Realty': 'KIM', 'Kinder Morgan': 'KMI', 'KLA Corporation': 'KLAC', 'Kraft Heinz': 'KHC', 'Kroger': 'KR',
              'L3Harris': 'LHX'}
nasdaq_dict = {'Adobe Inc.': 'ADBE', 'ADP': 'ADP', 'Airbnb': 'ABNB', 'GOOGLE': 'GOOGL', 'GOOGLE': 'GOOG', 'Amazon': 'AMZN',
    'Advanced Micro Devices Inc.': 'AMD', 'American Electric Power': 'AEP', 'Amgen': 'AMGN', 'Analog Devices': 'ADI', 'Ansys': 'ANSS', 'Apple Inc.': 'AAPL',
    'Applied Materials': 'AMAT', 'ASML Holding': 'ASML', 'AstraZeneca': 'AZN', 'Atlassian': 'TEAM', 'Autodesk': 'ADSK', 'Baker Hughes': 'BKR',
    'Biogen': 'BIIB', 'Booking Holdings': 'BKNG', 'Broadcom Inc.': 'AVGO', 'Cadence Design Systems': 'CDNS', 'CDW Corporation': 'CDW',
    'Charter Communications': 'CHTR', 'Cintas': 'CTAS', 'Cisco': 'CSCO', 'Coca-Cola Europacific Partners': 'CCEP', 'Cognizant': 'CTSH', 'Comcast': 'CMCSA',
    'Constellation Energy': 'CEG', 'Copart': 'CPRT', 'CoStar Group': 'CSGP', 'Costco': 'COST', 'CrowdStrike': 'CRWD', 'CSX Corporation': 'CSX',
    'Datadog': 'DDOG', 'DexCom': 'DXCM', 'Diamondback Energy': 'FANG', 'Dollar Tree': 'DLTR', 'DoorDash': 'DASH', 'Electronic Arts': 'EA',
    'Exelon': 'EXC', 'Fastenal': 'FAST', 'Fortinet': 'FTNT', 'GE HealthCare': 'GEHC', 'Gilead Sciences': 'GILD', 'GlobalFoundries': 'GFS',
    'Honeywell': 'HON', 'Idexx Laboratories': 'IDXX', 'Illumina, Inc.': 'ILMN', 'Intel': 'INTC', 'Intuit': 'INTU', 'Intuitive Surgical': 'ISRG',
    'Keurig Dr Pepper': 'KDP', 'KLA Corporation': 'KLAC', 'Kraft Heinz': 'KHC', 'Lam Research': 'LRCX', 'Lululemon': 'LULU', 'Marriott International': 'MAR',
    'Marvell Technology': 'MRVL', 'MercadoLibre': 'MELI', 'Meta Platforms': 'META', 'Microchip Technology': 'MCHP', 'Micron Technology': 'MU', 'Microsoft': 'MSFT',
    'Moderna': 'MRNA', 'Mondelēz International': 'MDLZ', 'MongoDB Inc.': 'MDB', 'Monster Beverage': 'MNST', 'Netflix': 'NFLX', 'Nvidia': 'NVDA', 'NXP': 'NXPI',
    'O\'Reilly Automotive': 'ORLY', 'Old Dominion Freight Line': 'ODFL', 'Onsemi': 'ON', 'Paccar': 'PCAR', 'Palo Alto Networks': 'PANW', 'Paychex': 'PAYX',
    'PayPal': 'PYPL', 'PDD Holdings': 'PDD', 'PepsiCo': 'PEP', 'Qualcomm': 'QCOM', 'Regeneron': 'REGN', 'Roper Technologies': 'ROP', 'Ross Stores': 'ROST',
    'Sirius XM': 'SIRI', 'Splunk': 'SPLK', 'Starbucks': 'SBUX', 'Synopsys': 'SNPS', 'Take-Two Interactive': 'TTWO', 'T-Mobile US': 'TMUS', 'Tesla, Inc.': 'TSLA',
    'Texas Instruments': 'TXN', 'The Trade Desk': 'TTD', 'Verisk': 'VRSK', 'Vertex Pharmaceuticals': 'VRTX', 'Walgreens Boots Alliance': 'WBA',
    'Warner Bros. Discovery': 'WBD', 'Workday, Inc.': 'WDAY', 'Xcel Energy': 'XEL', 'Zscaler': 'ZS',
}
commodities_dict = {"Brent Crude Oil": "BZ=F", "Cocoa": "CC=F", "Coffee": "KC=F", "Copper": "HG=F", 
    "Corn Futures": "ZC=F", "Cotton": "CT=F", "Heating Oil": "HO=F", "KC HRW Wheat Futures": "KE=F", 
    "Lean Hogs Futures": "HE=F", "Live Cattle Futures": "LE=F", "Mont Belvieu LDH Propane (OPIS)": "B0=F", 
    "Natural Gas": "NG=F", "Orange Juice": "OJ=F", "OURO": "GC=F", "Oat Futures": "ZO=F", 
    "Palladium": "PA=F", "PETROLEO CRU": "CL=F", "Platinum": "PL=F", "RBOB Gasoline": "RB=F", 
    "Random Length Lumber Futures": "LBS=F", "Rough Rice Futures": "ZR=F", "Silver": "SI=F", 
    "Soybean Futures": "ZS=F", "Soybean Oil Futures": "ZL=F", "S&P Composite 1500 ESG Tilted I": "ZM=F", 
    "Sugar #11": "SB=F", "WisdomTree International High D": "GF=F"
}


assets_list = {'CURRENCIES': currencies_dict, 
               'CRYPTO': crypto_dict, 
               'B3_STOCKS': b3_stocks, 
               'SP500': sp500_dict, 
               'NASDAQ100': nasdaq_dict,
               'indexes': indexes_dict}

# combining dictionaries when the user selects one or more in assets_list
selected_dict_names = st.multiselect('Select dictionaries to combine', list(assets_list.keys()))
combined_dict = {}
for name in selected_dict_names:
    dictionary = assets_list.get(name)
    if dictionary:
        combined_dict.update(dictionary)

# dictionary to actually store retrieved data
selected_ticker_dict = {}

# looping through the chosen tickers
if selected_dict_names:
    # the list to iterate over tickers
    tickers = st.multiselect('Asset Selection', list(combined_dict.keys()))
    if tickers and st.button("Download data"):
        for key in tickers:
            if key in combined_dict:
                selected_ticker_dict[key] = combined_dict[key]
        # Assigning data object as the result of the function download_data
        session_state.data = download_data(selected_ticker_dict, selected_timeframes)

#### RE-MUESTREO ####

frequency = {
        'Daily': 'D',
        'Weekly': 'W',
        'Quaterly': '2W',
        'Monthly': 'M',
        'Bimonthly': '2M',
        'Quarterly': '3M',
        'Four-monthly': '4M',
        'Semiannual': '6M',
        'Annual': 'A'
    }	

# moving average for NaN ocurrencies
st.sidebar.markdown('**Moving Avarage**')
moving_avg_days =  st.sidebar.number_input('Day(s):',1, 100, 3,step=1)                     
method = st.sidebar.selectbox("Method:", ['gap', 'rolling'])

if st.sidebar.button("Apply") and session_state.data is not None:
	session_state.data = fill_moving_avg(session_state.data, moving_avg_days, method)       

st.sidebar.markdown('**Missing Values**')    
remove_nan = st.sidebar.button('Dropna')
if remove_nan:
    session_state.data = session_state.data.dropna(axis = 1)

if session_state.data is not None:
    st.markdown(f'**Total of missing entries:** {session_state.data.isna().sum().sum()}')
    st.dataframe(session_state.data.isna().sum().to_frame().T)
    st.dataframe(session_state.data)
    tickers = [str(col).split("_")[0] for col in session_state.data.columns]
    tickers  = set(tickers)
    tickers_df = pd.DataFrame(tickers, columns=['Tickers'])

###############################################################################################################

if session_state.df is not None:
   st.dataframe(session_state.df)
   st.subheader('Optimization')
   close_price_data = [col for col  in session_state.data.columns if col.endswith('_Close')]
   session_state.portfolio  = session_state.data[close_price_data]

if session_state.portfolio is not None and not session_state.portfolio.empty:
    df, optimal_weights = EfficientFrontier(session_state.portfolio)
    corr_and_cov(session_state.portfolio)
    plot_cum_returns(session_state.portfolio, optimal_weights)

else:
    st.warning("Portfolio is empty, please download the data first.")


