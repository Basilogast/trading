import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timezone, timedelta
import time

class FindingTradeMACDRSI(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, ema_s, ema_l, signal_mw, periods, rsi_upper, rsi_lower):
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
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.signal_mw = signal_mw
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

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
        # MACD logic
        df["EMA_S"] = df[self.instrument].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        df["MACD"] = df["EMA_S"] - df["EMA_L"]
        df["MACD_Signal"] = df["MACD"].ewm(span=self.signal_mw, min_periods=self.signal_mw).mean()
        macd_signal = np.where(df["MACD"] - df["MACD_Signal"] > 0, 1, -1)
        # RSI logic
        df["returns"] = df[self.instrument].diff()
        df["U"] = np.where(df["returns"] > 0, df["returns"], 0)
        df["D"] = np.where(df["returns"] < 0, -df["returns"], 0)
        df["MA_U"] = df["U"].rolling(self.periods).mean()
        df["MA_D"] = df["D"].rolling(self.periods).mean()
        df["RSI"] = df["MA_U"] / (df["MA_U"] + df["MA_D"]) * 100
        rsi_signal = np.where(df["RSI"] > self.rsi_upper, -1, np.nan)
        rsi_signal = np.where(df["RSI"] < self.rsi_lower, 1, rsi_signal)
        rsi_signal = pd.Series(rsi_signal, index=df.index).fillna(0)
        # Combined logic: only trade when both signals agree
        df["position_MACD"] = macd_signal
        df["position_RSI"] = rsi_signal.astype(int)
        df["position"] = np.where(df["position_MACD"] == df["position_RSI"], df["position_MACD"], 0)
        df["position"] = pd.Series(df["position"], index=df.index).ffill().fillna(0)
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
class FindingTradeMACD(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, ema_s, ema_l, signal_mw):
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
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.signal_mw = signal_mw

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
        # MACD logic
        df["EMA_S"] = df[self.instrument].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        df["MACD"] = df["EMA_S"] - df["EMA_L"]
        df["MACD_Signal"] = df["MACD"].ewm(span=self.signal_mw, min_periods=self.signal_mw).mean()
        df["position"] = np.where(df["MACD"] - df["MACD_Signal"] > 0, 1, -1)
        df["position"] = df["position"].ffill().fillna(0)
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
class FindingTradeEMARSI(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, ema_s, ema_l, periods, rsi_upper, rsi_lower):
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
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

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
        # EMA logic
        df["EMA_S"] = df[self.instrument].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        ema_signal = np.where(df["EMA_S"] > df["EMA_L"], 1, -1)
        # RSI logic
        df["returns"] = df[self.instrument].diff()
        df["U"] = np.where(df["returns"] > 0, df["returns"], 0)
        df["D"] = np.where(df["returns"] < 0, -df["returns"], 0)
        df["MA_U"] = df["U"].rolling(self.periods).mean()
        df["MA_D"] = df["D"].rolling(self.periods).mean()
        df["RSI"] = df["MA_U"] / (df["MA_U"] + df["MA_D"]) * 100
        rsi_signal = np.where(df["RSI"] > self.rsi_upper, -1, np.nan)
        rsi_signal = np.where(df["RSI"] < self.rsi_lower, 1, rsi_signal)
        rsi_signal = pd.Series(rsi_signal, index=df.index).fillna(0)
        # Combined logic: only trade when both signals agree
        df["position"] = np.where(ema_signal == rsi_signal, ema_signal, 0)
        df["position"] = pd.Series(df["position"], index=df.index).ffill().fillna(0)
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

class FindingTradeRSI(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, periods, rsi_upper, rsi_lower):
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
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

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
        # RSI logic
        df["returns"] = df[self.instrument].diff()
        df["U"] = np.where(df["returns"] > 0, df["returns"], 0)
        df["D"] = np.where(df["returns"] < 0, -df["returns"], 0)
        df["MA_U"] = df["U"].rolling(self.periods).mean()
        df["MA_D"] = df["D"].rolling(self.periods).mean()
        df["RSI"] = df["MA_U"] / (df["MA_U"] + df["MA_D"]) * 100
        # Signal logic
        rsi_signal = np.where(df["RSI"] > self.rsi_upper, -1, np.nan)
        rsi_signal = np.where(df["RSI"] < self.rsi_lower, 1, rsi_signal)
        df["position"] = pd.Series(rsi_signal, index=df.index).fillna(0)
        df["position"] = df["position"].ffill().fillna(0)
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

class FindingTradeEMA(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, ema_s, ema_l):
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
        self.ema_s = ema_s
        self.ema_l = ema_l

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
        # EMA crossover logic
        df["EMA_S"] = df[self.instrument].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        df["position"] = np.where(df["EMA_S"] > df["EMA_L"], 1, -1)
        df["position"] = df["position"].ffill().fillna(0)
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