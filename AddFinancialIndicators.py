import ta
import pandas as pd

# technical analysis library: 
# 	- https://technical-analysis-library-in-python.readthedocs.io/en/latest/

#comment out whichever dataset is to not be used and change new_data to respective value
msft = pd.read_csv("data/stock-time-series/MSFT_2006-01-01_to_2018-01-01.csv") #training data
#msft = pd.read_csv("data/MSFT_New/MSFT_new.csv") #new data
new_data = False #set true if new data

'''
Momentum indicator: Money Flow Index 
	Uses both price and volume to measure buying and selling 
	pressure. It is positive when the typical price rises 
	(buying pressure) and negative when the typical price 
	declines (selling pressure). A ratio of positive and 
	negative money flow is then plugged into an RSI formula to 
	create an oscillator that moves between zero and one hundred.
'''
mfi = ta.momentum.money_flow_index(msft.High,
								   msft.Low,
								   msft.Close,
								   msft.Volume,
								   n=14,
								   fillna=False)

'''
Momentum indicator: Relative Strength Index
	Compares the magnitude of recent gains and losses over a 
	specified time period to measure speed and change of 
	price movements of a security. It is primarily used to 
	attempt to identify overbought or oversold conditions in 
	the trading of an asset.
'''
rsi = ta.momentum.rsi(msft.Close,
					  n=13,
					  fillna=False)

'''
Momentum Indicator: Stochastic Oscillator
	Developed in the late 1950s by George Lane. The stochastic 
	oscillator presents the location of the closing price of a 
	stock in relation to the high and low range of the price 
	of a stock over a period of time, typically a 14-day period.
'''
so = ta.momentum.stoch(msft.High,
					   msft.Low,
					   msft.Close,
					   n=14,
					   fillna=False)

'''
Volume Indicator: Chaikin Money Flow (CMF)
	It measures the amount of Money Flow Volume over a 
	specific period.
'''
cmf = ta.volume.chaikin_money_flow(msft.High,
								   msft.Low,
								   msft.Close,
								   msft.Volume,
								   n=14,
								   fillna=False)

'''
Volatility Indicator: Average True Range (ATR)
	The indicator provide an indication of the degree of 
	price volatility. Strong moves, in either direction, 
	are often accompanied by large ranges, or large True 
	Ranges.
'''  
atr = ta.volatility.average_true_range(msft.High,
									   msft.Low,
									   msft.Close,
									   n=14,
									   fillna=False)

'''
Volatility Indiciator: Bollinger Bands (BB)
	N-period simple moving average (MA).
'''
bb = ta.volatility.bollinger_mavg(msft.Close,
								  n=14,
								  fillna=False)

'''
Trend Indicator: Average Directional Movement Index (ADX)
	The Plus Directional Indicator (+DI) and Minus Directional 
	Indicator (-DI) are derived from smoothed averages of 
	these differences, and measure trend direction over time. 
	These two indicators are often referred to collectively as 
	the Directional Movement Indicator (DMI).

	The Average Directional Index (ADX) is in turn derived from 
	the smoothed averages of the difference between +DI and -DI, 
	and measures the strength of the trend (regardless of 
	direction) over time.

	Using these three indicators together, chartists can 
	determine both the direction and strength of the trend.
'''
adx = ta.trend.adx(msft.High,
				   msft.Low,
				   msft.Close,
				   n=14,
				   fillna=False)

'''
Trend Indicator: Aroon Indicator (AI)
	Identify when trends are likely to change direction 
	(uptrend).

	Aroon Up = ((N - Days Since N-day High) / N) x 100
'''
ai = ta.trend.aroon_up(msft.Close, 
					   n=14, 
					   fillna=False)


#save indicators to csv files
#NOTE: may need to delete "" from top of csv files
path = 'data/financial-indicators/'

if new_data: 
	mfi.to_csv(path+'mfi_new.csv', index=False)
	rsi.to_csv(path+'rsi_new.csv', index=False)
	so.to_csv(path+'so_new.csv', index=False)
	cmf.to_csv(path+'cmf_new.csv', index=False)
	atr.to_csv(path+'atr_new.csv', index=False)
	bb.to_csv(path+'bb_new.csv', index=False)
	adx.to_csv(path+'adx_new.csv', index=False)
	ai.to_csv(path+'ai_new.csv', index=False)
else: 
	mfi.to_csv(path+'mfi.csv', index=False)
	rsi.to_csv(path+'rsi.csv', index=False)
	so.to_csv(path+'so.csv', index=False)
	cmf.to_csv(path+'cmf.csv', index=False)
	atr.to_csv(path+'atr.csv', index=False)
	bb.to_csv(path+'bb.csv', index=False)
	adx.to_csv(path+'adx.csv', index=False)
	ai.to_csv(path+'ai.csv', index=False)

