from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import os
import csv
import logging
from dotenv import load_dotenv
import telegram
import ccxt

# Ortam değişkenlerini yükle
from pathlib import Path
load_dotenv(dotenv_path=Path('.') / '.env')
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ✅ Burada kontrol yapıyoruz
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("❌ TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID bulunamadı! .env dosyasını kontrol et.")

tg_bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Logging yapılandırması
logging.basicConfig(
    filename='bot_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def fetch_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=11, multiplier=1.7)['SUPERT_11_1.7']
        df['trend'] = df['supertrend'] < df['close']
        df['trend_shift'] = df['trend'].shift(1)
        df['trend_change'] = df['trend'] != df['trend_shift']
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_spike'] = df['volume'] > df['volume_sma'] * 2
        df['volatility'] = ((df['high'] - df['low']) / df['open']) * 100
        return df.dropna()
    except Exception as e:
        logging.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()

# Uygulama ayarları
REFRESH_INTERVAL = 10  # saniye
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
LIMIT = 100
exchange = ccxt.okx({
    'apiKey': os.getenv("OKX_API_KEY"),
    'secret': os.getenv("OKX_SECRET"),
    'password': os.getenv("OKX_PASSPHRASE"),
    'enableRateLimit': True
})

active_position = False
BALINA_FIYAT_YUZDESI = 0.4  # %0.4 değişim
BALINA_SURE = 3  # Kaç bar önceye göre karşılaştırma
RSI_ARALIKLARI = [(45, 55), (48, 60), (50, 62), (52, 63), (55, 65), (58, 68)]

import asyncio

def send_telegram_message(message):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
        else:
            loop.run_until_complete(tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
    except Exception as e:
        logging.error(f"Telegram mesaj hatası: {e}")

def wundertrading_sinyal_gonder(tip, fiyat, miktar):
    try:
        payload = {
            "pair": SYMBOL,
            "order_type": "market",
            "side": tip.lower(),
            "amount": miktar,
            "price": fiyat
        }
        response = requests.post("https://api.wundertrading.com/api/v1/signal/v2", json=payload)
        logging.info(f"WunderTrading sinyali gönderildi: {payload}")
        send_telegram_message(f"📡 WunderTrading sinyali gönderildi: {tip.upper()} - {miktar:.4f} BTC @ {fiyat:.2f}")
        return response.status_code == 200
    except Exception as e:
        logging.error(f"WunderTrading sinyal hatası: {e}")
        return False

def execute_order(order_type, amount):
    try:
        if order_type == 'BUY':
            order = exchange.create_market_buy_order(SYMBOL, amount)
        elif order_type == 'SELL':
            order = exchange.create_market_sell_order(SYMBOL, amount)

        fiyat = exchange.fetch_ticker(SYMBOL)['last']

        send_telegram_message(
            f"📢 {order_type} EMRİ GERÇEKLEŞTİ\nMiktar: {amount:.6f} BTC\nFiyat: {fiyat:.2f} USDT"
        )

        return order
    except Exception as e:
        send_telegram_message(f"❌ Emir Hatası: {str(e)}")
        return None

active_position = False
app = Dash(__name__)

app.layout = html.Div([
    html.H1('BTC/USDT Canlı Grafik Paneli', style={'textAlign': 'center'}),
    dcc.Graph(id='live-graph', style={'height': '80vh'}),
    html.Div(id='pnl-output', style={'textAlign': 'center', 'fontSize': 18, 'padding': '10px'}),
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL * 1000, n_intervals=0)
])

def uygula_kar_alma_ve_stoploss(latest_price):
    global trades, pnl, active_position, global_df

    if not trades:
        return

    # En son alış pozisyonlarını bul
    alimlar = [t for t in trades if t['type'] == 'BUY']
    if not alimlar:
        return

    # Ortalama maliyet hesapla
    toplam_miktar = sum(t['amount'] for t in alimlar)
    toplam_fiyat = sum(t['price'] * t['amount'] for t in alimlar)
    ort_maliyet = toplam_fiyat / toplam_miktar if toplam_miktar != 0 else 0

    # Kademeli kâr alma
    toplam_satim = 0
    for seviye, oran in zip(KAR_SEVIYELERI, SATIS_ORANLARI):
        hedef_fiyat = ort_maliyet * seviye
        if hedef_fiyat >= ort_maliyet and latest_price >= hedef_fiyat:
            miktar = toplam_miktar * oran
            result = execute_order('SELL', miktar)
            if result:
                send_telegram_message(f"✅ Kademeli Satım Gerçekleştirildi\n✔️ Fiyat: {latest_price:.2f} USDT\n✔️ Miktar: {miktar:.6f} BTC")
                trades.append({
                    'timestamp': datetime.utcnow(),
                    'type': 'SELL',
                    'price': latest_price,
                    'amount': miktar
                })
                toplam_satim += miktar

    # Eğer tüm miktar satıldıysa pozisyonu kapat
    if toplam_satim >= toplam_miktar * 0.99:  # Küçük sapmalar için tolerans
        active_position = False

    # Akıllı Stop
    rsi_son = global_df['rsi'].iloc[-1]
    hacim_son = global_df['volume'].iloc[-1]
    hacim_ort = global_df['volume'].rolling(10).mean().iloc[-1]

    if latest_price < ort_maliyet and rsi_son < RSI_STOP_ESIK and hacim_son > hacim_ort * HACIM_SPIKE_CARPANI:
        if latest_price >= ort_maliyet * STOP_ZARAR_ESIK:
            result = execute_order('SELL', toplam_miktar - toplam_satim)
            if result:
                send_telegram_message(f"🛑 STOP-LOSS Satımı Yapıldı!\nFiyat: {latest_price:.2f} USDT\nTetikleyen: RSI < {RSI_STOP_ESIK} & Hacim Spike")
                trades.append({
                    'timestamp': datetime.utcnow(),
                    'type': 'SELL',
                    'price': latest_price,
                    'amount': toplam_miktar - toplam_satim
                })
                active_position = False
                pnl += (latest_price - ort_maliyet) * (toplam_miktar - toplam_satim)

@app.callback(
    [Output('live-graph', 'figure'), Output('pnl-output', 'children')],
    [Input('interval-component', 'n_intervals')])

def update_graph(n):
    print("🟡 update_graph başladı")  # DEBUG
    global trades, pnl, position, entry_price, highest_price, trailing_stop, active_position, global_df
    trades = []
    pnl = 0.0
    position = None
    entry_price = 0.0
    highest_price = 0.0
    trailing_stop = None

    global_df = fetch_data()
    if global_df.empty:
        return go.Figure(), "Veri alınamadı."

    global_df['price_delta'] = global_df['close'].pct_change(periods=BALINA_SURE) * 100
    global_df['balina_hareketi'] = (global_df['price_delta'].abs() > BALINA_FIYAT_YUZDESI) & global_df['volume_spike']
    global_df['rsi_up'] = global_df['rsi'] > global_df['rsi'].shift(1)

    rsi_deger = global_df['rsi'].iloc[-1]
    volatilite_deger = global_df['volatility'].iloc[-1]

    def kademe_belirle():
        if volatilite_deger > 2.5:
            return [0.5, 0.3, 0.2] if rsi_deger < 50 else [0.4, 0.35, 0.25]
        elif volatilite_deger > 1.5:
            return [0.35, 0.35, 0.3] if rsi_deger < 50 else [0.3, 0.4, 0.3]
        else:
            return [0.25, 0.35, 0.4] if rsi_deger < 50 else [0.2, 0.4, 0.4]

    KADEMELER = kademe_belirle()

    en_iyi_rsi_araligi = (51, 63)  # optimize edilmiş
    en_yuksek_kazanc = -float('inf')

    for alt, ust in RSI_ARALIKLARI:
        temp_df = global_df.copy()
        temp_df['buy_signal'] = (
            (temp_df['trend']) &
            (temp_df['rsi'].between(alt, ust)) &
            (temp_df['rsi_up']) &
            (temp_df['close'] > temp_df['supertrend'])
        )

        temp_df['buy_signal'] |= (
            (temp_df['balina_hareketi']) &
            (temp_df['trend']) &
            (temp_df['close'] > temp_df['supertrend'])
        )

        temp_df['sell_signal'] = (
            (temp_df['trend_change']) &
            (~temp_df['trend']) &
            (temp_df['rsi'] < 45) &
            (temp_df['close'] < temp_df['supertrend'])
        )

        temp_trades = []
        position = None
        entry_price = 0.0
        gain = 0.0

        for i in range(len(temp_df)):
            row = temp_df.iloc[i]
            if row['buy_signal'] and position is None:
                entry_price = row['close']
                position = 1
            elif row['sell_signal'] and position == 1:
                gain += row['close'] - entry_price
                position = None

        if gain > en_yuksek_kazanc:
            en_yuksek_kazanc = gain
            en_iyi_rsi_araligi = (alt, ust)

        # 🔽 Yeni alım stratejisini buraya ekle
        alt, ust = en_iyi_rsi_araligi
        volume_sma = global_df['volume'].rolling(window=10).mean()
        global_df['volume_spike'] = global_df['volume'] > volume_sma * 1.3
        global_df['rsi_up'] = global_df['rsi'] > global_df['rsi'].shift(1)
        global_df['fiyat_artiyor'] = global_df['close'] > global_df['close'].shift(1)

        global_df['buy_signal'] = (
            global_df['rsi_up'] &
            global_df['fiyat_artiyor'] &
            global_df['volume_spike'] &
            (global_df['close'] > global_df['supertrend'] * 0.995)
        )

        global_df['buy_signal'] |= (
            (global_df['balina_hareketi']) &
            (global_df['trend']) &
            (global_df['close'] > global_df['supertrend'] * 0.995)
        )


        global_df['sell_signal'] = (
            (global_df['trend_change']) &
            (~global_df['trend']) &
            (global_df['rsi'] < 45) &
            (global_df['close'] < global_df['supertrend'])
        )

    is_uptrend = global_df['trend'].iloc[-1]
    is_downtrend = not is_uptrend
    latest_price = global_df['close'].iloc[-1]
    uygula_kar_alma_ve_stoploss(latest_price)
    TOPLAM_USDT = 10

    def kademeli_islem_yap(tip):
        for oran in KADEMELER:
            miktar = (TOPLAM_USDT * oran) / latest_price
            result = execute_order(tip, miktar)
            if result:
                timestamp = datetime.utcnow()
                with open('islemler.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, tip, latest_price, miktar])

                trades.append({
                    'timestamp': timestamp,
                    'type': tip,
                    'price': latest_price,
                    'amount': miktar
                })

    if (global_df['buy_signal'].iloc[-1] or global_df['buy_signal'].iloc[-2]) and not active_position and is_uptrend:
        print("🎯 Alım sinyali oluştu!")
        kademeli_islem_yap('BUY')

    elif global_df['sell_signal'].iloc[-1] and active_position and is_downtrend:
        print("📉 Satım sinyali oluştu!")
        kademeli_islem_yap('SELL')

    trades = []
    pnl = 0.0
    position = None
    entry_price = 0.0
    highest_price = 0.0
    trailing_stop = None
    active_position = False  # 🔄 En son pozisyon bilgisi CSV'den okunmadan önce sıfırlanır

    if os.path.exists('islemler.csv'):
        try:
            with open('islemler.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 4:
                        timestamp, trade_type, price, amount = row
                        price = float(price)
                        amount = float(amount)
                        timestamp = pd.to_datetime(timestamp)
                        trades.append({
                            'timestamp': timestamp,
                            'type': trade_type,
                            'price': price,
                            'amount': amount
                        })

                        if trade_type == 'BUY':
                            position = amount
                            entry_price = price
                            highest_price = price
                            trailing_stop = price * (1 - 0.015)
                            active_position = True
                        elif trade_type == 'SELL' and position:
                            pnl += (price - entry_price) * position
                            position = None
                            trailing_stop = None
                            active_position = False
        except Exception as e:
            print(f"CSV okuma hatası: {e}")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=global_df['timestamp'],
                                 open=global_df['open'],
                                 high=global_df['high'],
                                 low=global_df['low'],
                                 close=global_df['close'],
                                 name='BTC/USDT'))

    fig.add_trace(go.Scatter(x=global_df['timestamp'],
                             y=global_df['supertrend'],
                             mode='lines',
                             name='Supertrend'))

    fig.add_trace(go.Scatter(x=global_df[global_df['buy_signal']]['timestamp'],
                             y=global_df[global_df['buy_signal']]['close'],
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='green', size=10),
                             name='Alım'))
    fig.add_trace(go.Scatter(
        x=[t['timestamp'] for t in trades if t['type'] == 'BUY'],
        y=[t['price'] for t in trades if t['type'] == 'BUY'],
        mode='markers',
        marker=dict(symbol='circle', color='green', size=8),
        name='Gerçek Alımlar'))

    fig.add_trace(go.Scatter(x=global_df[global_df['sell_signal']]['timestamp'],
                             y=global_df[global_df['sell_signal']]['close'],
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Satım'))
    fig.add_trace(go.Scatter(
        x=[t['timestamp'] for t in trades if t['type'] == 'SELL'],
        y=[t['price'] for t in trades if t['type'] == 'SELL'],
        mode='markers',
        marker=dict(symbol='circle', color='red', size=8),
        name='Gerçek Satışlar'))

    if active_position and trailing_stop:
        fig.add_trace(go.Scatter(x=[global_df['timestamp'].iloc[-1]] * 2,
                                 y=[trailing_stop, trailing_stop],
                                 mode='lines',
                                 line=dict(color='red', width=2, dash='dash'),
                                 name='Trailing Stop'))

    status_text = f"Toplam PnL (kâr/zarar): {pnl:.2f} USDT"
    if active_position:
        status_text += " | 🟢 Pozisyon Aktif"
    else:
        status_text += " | 🔴 Pozisyon Kapalı"

    print("🟢 update_graph tamamlandı")  # DEBUG
    return fig, status_text


KAR_SEVIYELERI = [1.012, 1.025, 1.04]  # Kâr alma seviyeleri
SATIS_ORANLARI = [0.4, 0.35, 0.25]     # Her seviyede satılacak oranlar
STOP_ZARAR_ESIK = 0.985               # Ortalama maliyetin %1.5 altı stop eşiği
RSI_STOP_ESIK = 40                    # RSI stop eşiği
HACIM_SPIKE_CARPANI = 1.8             # Hacim spike çarpanı


if __name__ == '__main__':
    app.run(debug=True)

