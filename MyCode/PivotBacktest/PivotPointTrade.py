import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timezone, timedelta
import time

class PivotPointTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        
        #*****************add strategy-specific attributes here******************
        #************************************************************************
    
    def get_most_recent(self, days=5):
        while True:
            time.sleep(2)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            now = now - timedelta(microseconds=now.microsecond)
            past = now - timedelta(days=days)
            df = self.get_history(instrument=self.instrument, start=past, end=now,
                                  granularity="S5", price="M", localize=False).c.dropna().to_frame()
            df.rename(columns={"c": self.instrument}, inplace=True)
            df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.now(timezone.utc)) - self.last_bar < self.bar_length:
                break
                
    def on_success(self, time, bid, ask):
        print(self.ticks, end=" ")
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument: (ask + bid) / 2}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        self.raw_data = pd.concat([
            self.raw_data,
            self.tick_data.resample(self.bar_length, label="right").last().ffill().iloc[:-1]
        ])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self):
        df = self.raw_data.copy()
        #******************** define your strategy here ************************
        # Calculate daily OHLC
        ohlc_daily = df[self.instrument].resample('D').agg(['first', 'max', 'min', 'last']).dropna()
        ohlc_daily.columns = ['Open_d', 'High_d', 'Low_d', 'Close_d']
        # Merge daily OHLC into intraday data (shifted by 1 day)
        df = pd.concat([df, ohlc_daily.shift().reindex(df.index, method='ffill')], axis=1)
        df.dropna(inplace=True)
        # Calculate Pivot Point and S/R levels
        df['PP'] = (df['High_d'] + df['Low_d'] + df['Close_d']) / 3
        df['S1'] = df['PP'] * 2 - df['High_d']
        df['R1'] = df['PP'] * 2 - df['Low_d']
        # Pivot Point logic:
        # 1 = long, -1 = short, 0 = neutral
        df['position'] = 0
        cond_long = (df[self.instrument] > df['PP']) & (df[self.instrument] < df['R1'])
        cond_short = (df[self.instrument] < df['PP']) & (df[self.instrument] > df['S1'])
        df.loc[cond_long, 'position'] = 1
        df.loc[cond_short, 'position'] = -1
        df['position'] = df['position'].ffill().fillna(0)
        #***********************************************************************
        self.data = df.copy()
    
    def execute_trades(self):
        pos = self.data["position"].iloc[-1]
        if pos == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif pos == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif pos == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print(f"{time} | {going}")
        print(f"{time} | units = {units} | price = {price} | P&L = {pl} | Cum P&L = {cumpl}")
        print(100 * "-" + "\n")
