#STAT454: Machine Learning - Project 

setwd("~/Desktop/STAT454/Project")

# before running, generate financial indicators in .py file
# using MSFT_2006-01-01_to_2018-01-01.csv. see .py file for 
# instructions
msft = processData(newData=FALSE)
# before running new data generate new financial indicators in .py file
# using MSFT_new.csv. see .py file for instructions
msft_new = processData(newData=TRUE)

# function processes data into desired form for machine learning
#   Params:
#     - newData = FALSE if processing training/testing dat and 
#               = TRUE if processing 30 days worth of recent data
processData = function(newData = FALSE) {

  if (!newData) msft = read.csv("data/stock-time-series/MSFT_2006-01-01_to_2018-01-01.csv")
  if (newData) msft = read.csv("data/MSFT_New/MSFT_new.csv") #2018-09-24 ... 2018-11-23

  #add response variable (stock price increase next day excluding first 13 days for indicators)
  increased <- NULL
  for (i in 1:(length(msft[,1])-1)) {
    if (i %in% c(1:13)) {
      increased[i] = NA
    } else if (msft[i,5] < msft[i+1,5]) {
      increased[i] = 1
    } else {
      increased[i] = 0
    }
  }
  increased[length(msft[,1])] = NA
  
  msft$Increase = increased
  msft$Increase = as.factor(msft$Increase)

  if (!newData) {
    #incorporate Google search term "microsoft stock" trend to feature set
    #   Interest over time
    #       Numbers represent search interest relative to the highest point 
    #       on the chart for the given region and time. A value of 100 is 
    #       the peak popularity for the term. A value of 50 means that the 
    #       term is half as popular. A score of 0 means there was not enough 
    #       data for this term.
    #   For future prediction: 
    #       Take average of past days (in month) trend values 
    
    msft_google_trend = read.csv("data/google-trends-msft.csv")
    msft_google_trend = msft_google_trend[-c(1:25),] #remove dates earlier than 2006-01-01
    month_ind = c(rep(c("01","02","03","04", "05", "06", "07", "08", "09", "10", "11", "12"), 
                      length(msft_google_trend[,1])/12 + 1))
    month_ind = month_ind[-length(month_ind)]
    year_ind = c(rep("2006", 12), rep("2007", 12), rep("2008", 12), rep("2009",12), 
                 rep("2010", 12), rep("2011", 12), rep("2012", 12), rep("2013", 12), 
                 rep("2014", 12), rep("2015", 12), rep("2016", 12), rep("2017", 12), 
                 rep("2018", 12))
    year_ind = year_ind[-length(year_ind)]
    
    msft_google_trend$MonthInd = month_ind
    msft_google_trend$YearInd = year_ind
    
    msft$YearInd = substr(msft$Date, 1, 4) 
    msft$MonthInd = substr(msft$Date, 6, 7) 
    
    msft_google_trend$trend = as.numeric(as.character(msft_google_trend$X))
    
    trend = NULL
    for (i in 1:length(msft[,1])) {
      for (j in 1:length(msft_google_trend[,1])) {
        if (msft$YearInd[i] == msft_google_trend$YearInd[j] &&
            msft$MonthInd[i] == msft_google_trend$MonthInd[j]) {
          trend[i] = msft_google_trend$trend[j]
        }
      }
    }
  } else if (newData) {
    msft$YearInd = substr(msft$Date, 1, 4) #run if new
    msft$MonthInd = substr(msft$Date, 6, 7)
    
    nov_trend = 34.13636364 #calculated in excel (monthly average)
    oct_trend = 47.4516129 #calculated in excel (monthly average)
    sept_trend = 36.57142857 #calculated in excel (monthly average)
    
    trend = NULL
    trend[c(1:5)] = sept_trend
    trend[c(6:28)] = oct_trend
    trend[c(29:44)] = nov_trend
  }

  msft$GoogleTrend = trend
  
  #add daily return feature 
  msft$DailyReturn = msft$Close/msft$Open - 1 

  #remove name column
  if (!newData) msft = msft[ ,-7] #don't do for new data

  #log volume for better mapping to prediction target
  msft$LogVolume = log(msft$Volume)

  if (!newData) {
    #load financial indicator features (skip over if new data to next set)
    adx = read.csv("data/financial-indicators/adx.csv")
    ai = read.csv("data/financial-indicators/ai.csv")
    atr = read.csv("data/financial-indicators/atr.csv")
    bb = read.csv("data/financial-indicators/bb.csv")
    cmf = read.csv("data/financial-indicators/cmf.csv")
    mfi = read.csv("data/financial-indicators/mfi.csv")
    rsi = read.csv("data/financial-indicators/rsi.csv")
    so = read.csv("data/financial-indicators/so.csv")
  } else if (newData) {
    # new data (skip if not new data)
    adx = read.csv("data/financial-indicators/adx_new.csv")
    ai = read.csv("data/financial-indicators/ai_new.csv")
    atr = read.csv("data/financial-indicators/atr_new.csv")
    bb = read.csv("data/financial-indicators/bb_new.csv")
    cmf = read.csv("data/financial-indicators/cmf_new.csv")
    mfi = read.csv("data/financial-indicators/mfi_new.csv")
    rsi = read.csv("data/financial-indicators/rsi_new.csv")
    so = read.csv("data/financial-indicators/so_new.csv")
  }

  adx = adx[-c(1:13), ]
  atr = atr[-c(1:13), ]
  
  msft = msft[-length(msft[,1]), ] #remove last row 
  msft = na.omit(msft)
  
  msft$ADX = adx
  msft$AI = ai[ ,1]
  msft$ATR = atr
  msft$BB  = bb[ ,1]
  msft$CMF = cmf[ ,1]
  msft$MFI = mfi[ ,1]
  msft$RSI = rsi[ ,1]
  msft$SO = so[ ,1]
  
  #change date features to numeric
  msft$YearInd = as.numeric(msft$YearInd)
  msft$MonthInd = as.numeric(msft$MonthInd)
  
  #remove date column
  msft = msft[ ,-1]

  #save processed dataset 
  if (!newData) save(msft, file="data/msft.rda") #don't run if new
  if (newData) {
    #save processed dataset 
    msft_new = msft
    save(msft_new, file="data/msft_new.rda")
  }
  
  return(msft)
  
}
