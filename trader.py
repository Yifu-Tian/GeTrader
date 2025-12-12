# -*- coding: utf-8 -*-

import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:[your_proxy_port]'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:[your_proxy_port]'
import requests
import ccxt
import pandas as pd
import google.generativeai as genai 
from dotenv import load_dotenv
import time
from datetime import datetime
import feedparser
import json

# Load the environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

# 2. Fetch API keys 
api_key = os.getenv("GEMINI_API_KEY")
cmc_api_key = os.getenv("CMC_API_KEY")

if not api_key or not cmc_api_key:
    raise ValueError("API keys not found. Ensure '.env' has GEMINI_API_KEY and CMC_API_KEY.")

# 3. Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')
# ==========================================
# News Agent
# ==========================================
class NewsAgent:
    def get_latest_news(self):
        #  Google News RSS 
        rss_url = "https://news.google.com/rss/search?q=bitcoin+crypto+market+when:1h&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            
            headlines = []
            if feed.entries:
                # GET 3
                for entry in feed.entries[:3]:
                    headlines.append(f"- {entry.title} ({entry.published})")
                return "\n".join(headlines)
            else:
                return "No major news in the last hour."
        except Exception as e:
            return f"News fetch error: {e}"
        
# ==========================================
# Paper Trading Engine
# ==========================================
class PaperTradingEngine:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance     # 初始资金 (USDT)
        self.position = None               # 当前持仓: 'LONG', 'SHORT', None
        self.entry_price = 0.0             # 开仓价
        self.entry_time = None             # 开仓时间
        self.trade_history = []            # 交易历史
        self.strategy_score = 100          # 策略评分 (初始100)
        self.lessons_learned = []          # AI 的反思笔记

    def execute(self, action, price, reason):
        # 平仓逻辑 (如果有持仓，且信号相反或要求平仓)
        pnl = 0
        if self.position and (action == "CLOSE" or action != self.position):
            if self.position == "LONG":
                pnl = (price - self.entry_price) / self.entry_price * 100
            elif self.position == "SHORT":
                pnl = (self.entry_price - price) / self.entry_price * 100
            
            # 结算
            realized_pnl = self.balance * (pnl / 100)
            self.balance += realized_pnl
            
            # 记录历史
            trade_res = "WIN" if pnl > 0 else "LOSS"
            self.trade_history.append({
                "type": self.position,
                "entry": self.entry_price,
                "exit": price,
                "pnl_pct": pnl,
                "result": trade_res
            })
            
            # 策略评分调整
            if pnl > 0: self.strategy_score += 1
            else: self.strategy_score -= 2 # 亏损扣分更重，逼迫反思
            
            # 生成反思触发器
            reflection_trigger = True if pnl < -0.5 else False # 亏损超过0.5%就反思
            
            self.position = None # 仓位归零
            return f"平仓完成! PnL: {pnl:.2f}% ({trade_res})", reflection_trigger

        # 开仓逻辑
        if action in ["LONG", "SHORT"] and self.position is None:
            self.position = action
            self.entry_price = price
            self.entry_time = datetime.now()
            return f"开仓成功: {action} @ {price}", False

        return "持仓不动", False

    def get_status(self, current_price):
        floating_pnl = 0
        if self.position == "LONG":
            floating_pnl = (current_price - self.entry_price) / self.entry_price * 100
        elif self.position == "SHORT":
            floating_pnl = (self.entry_price - current_price) / self.entry_price * 100
            
        return floating_pnl

# ==========================================
# AI 分析
# ==========================================
def ask_evolutionary_ai(df, news_text, trader_engine, symbol):
    curr = df.iloc[-1]
    
    # 获取浮动盈亏
    floating_pnl = trader_engine.get_status(curr['close'])
    
    # 构建上下文
    context = f"""
    === 🌍 MACRO NEWS (Sentiment Analysis) ===
    {news_text}
    
    === 🏥 ACCOUNT HEALTH (Simulation) ===
    Balance: {trader_engine.balance:.2f} USDT
    Current Position: {trader_engine.position if trader_engine.position else 'EMPTY'}
    Entry Price: {trader_engine.entry_price}
    Floating PnL: {floating_pnl:.2f}%
    Strategy Score: {trader_engine.strategy_score}/100
    
    === 📒 STRATEGY MEMORY (Lessons form Past Mistakes) ===
    {chr(10).join(trader_engine.lessons_learned[-3:]) if trader_engine.lessons_learned else "No lessons yet. Starting fresh."}
    
    === 📊 TECHNICAL DATA ===
    Price: {curr['close']:.2f}
    RSI: {curr['RSI']:.2f}
    ATR: {curr['ATR']:.2f}
    Bollinger: {'Breakout Up' if curr['close'] > curr['Upper'] else ('Breakout Down' if curr['close'] < curr['Lower'] else 'Inside')}
    """

    prompt = f"""
    Role: You are an Evolutionary AI Trader. You learn from your own PnL.
    
    Task 1 (News Impact): Briefly evaluate if the news is Bullish/Bearish/Neutral for Crypto.
    Task 2 (Decision): Decide to LONG, SHORT, CLOSE, or HOLD.
    Task 3 (Self-Correction): 
    - If you are losing money (Floating PnL < -0.5%), admit your strategy was wrong and write a "Lesson".
    - If you are making money (Floating PnL > 1%), confirm your strategy is stable.
    
    Input Data:
    {context}
    
    Output Format:
    {{
        "news_sentiment": "Bullish/Bearish/Neutral",
        "action": "LONG/SHORT/CLOSE/HOLD",
        "reason": "Detailed technical + news reasoning...",
        "reflection": "If PnL is bad, write what you did wrong. If good, write what worked. If neutral, leave empty."
    }}
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        return {"action": "HOLD", "reason": f"AI Error: {str(e)}", "reflection": "", "news_sentiment": "Unknown"}
# ==========================================
# 数据处理
# ==========================================
class MarketDataHandler:
    def __init__(self):
        self.exchange = ccxt.okx({
            'proxies': {
                'http': f'http://127.0.0.1:[your_proxy_port]',
                'https': f'http://127.0.0.1:[your_proxy_port]',
            },
            'timeout': 30000,
        })

    def fetch_and_calculate(self, symbol="BTC/USDT", timeframe="5m"):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            if not bars: return None
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # --- 基础指标 ---
            df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean()
            df['EMA_25'] = df['close'].ewm(span=25, adjust=False).mean()
            
            # --- RSI ---
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            rs = gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + rs))

            # --- 布林带 (用于判断变盘) ---
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['Upper'] = df['SMA_20'] + (std * 2)
            df['Lower'] = df['SMA_20'] - (std * 2)
            
            # --- ATR (波动率 - 用于判断进攻还是防守) ---
            df['TR'] = df['high'] - df['low']
            df['ATR'] = df['TR'].rolling(window=14).mean()

            return df
        except Exception as e:
            print(f"数据获取失败: {e}")
            return None
# ==========================================
# 主程序
# ==========================================
def main():
    # init
    data_handler = MarketDataHandler()
    news_agent = NewsAgent()
    paper_trader = PaperTradingEngine()
    
    symbol = "BTC/USDT"
    print(f"\nAI 交易员已上线 | 初始资金: {paper_trader.balance} U")

    while True:
        try:
            # 1. 获取数据
            df = data_handler.fetch_and_calculate(symbol)
            if df is None: 
                time.sleep(10)
                continue
                
            # 2. 获取新闻
            news = news_agent.get_latest_news()
            
            # 3. AI 思考
            decision = ask_evolutionary_ai(df, news, paper_trader, symbol)
            
            # 4. 执行模拟交易
            current_price = df.iloc[-1]['close']
            exec_msg, needs_reflection = paper_trader.execute(decision['action'], current_price, decision['reason'])
            
            # 
            if decision['reflection']:
                paper_trader.lessons_learned.append(f"[{datetime.now().strftime('%H:%M')}] {decision['reflection']}")
                
            # 
            os.system('cls' if os.name == 'nt' else 'clear') 
            
            print("="*60)
            print(f"{datetime.now().strftime('%H:%M:%S')} | {symbol} : {current_price:.2f}")
            print(f"📰 新闻情绪: {decision['news_sentiment']}")
            print("-" * 60)
            print(f"决策: {decision['action']} | 策略评分: {paper_trader.strategy_score}")
            print(f"理由: {decision['reason']}")
            print("-" * 60)
            
            # 账户状态栏
            pnl = paper_trader.get_status(current_price)
            pnl_str = f"{pnl:+.2f}%"
            if pnl > 0: pnl_color = "🟩"
            elif pnl < 0: pnl_color = "🟥"
            else: pnl_color = "⬜"
            
            print(f"当前持仓: {paper_trader.position if paper_trader.position else '空仓'}")
            if paper_trader.position:
                print(f" 浮动盈亏: {pnl_color} {pnl_str} (入场: {paper_trader.entry_price})")
            
            print(f"账户余额: {paper_trader.balance:.2f} U")
            
            if paper_trader.lessons_learned:
                print("-" * 60)
                print(f"AI 进化笔记 (Latest):")
                print(f"{paper_trader.lessons_learned[-1]}")
            
            print("="*60)
            
            time.sleep(60)

        except KeyboardInterrupt:
            print("终止")
            break
        except Exception as e:
            print(f"Runtime Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
