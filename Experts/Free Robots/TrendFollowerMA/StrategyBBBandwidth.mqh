//=== StrategyBBBandwidth.mqh ===
#ifndef __StrategyBBBandwidth_mqh__
#define __StrategyBBBandwidth_mqh__

void EvaluateBB_Bandwidth(string sym, SymbolState* state) {
   if (SeriesInfoInteger(sym,ExpectedTimeframe,SERIES_SYNCHRONIZED)==false) return;
   if (PositionSelect(sym)) { HandleExit(sym,state); return; }
   if (TimeCurrent()-state.lastTradeTime<ReentryCooldownBars*PeriodSeconds(_Period)) return;
   
   double bid, ask;
   if (!SymbolInfoDouble(sym,SYMBOL_BID,bid)||!SymbolInfoDouble(sym,SYMBOL_ASK,ask)) return;
   
   double sma[];
   if (CopyBuffer(state.smaHandle,0,0,BBPeriod_Bandwidth+1,sma)<BBPeriod_Bandwidth+1) return;
   
   double atr[];
   if (CopyBuffer(state.atrHandle,0,0,1,atr)!=1||atr[0]<=0) return;
   
   int rsiHandle=iRSI(sym,ExpectedTimeframe,RSI_Period,PRICE_CLOSE);
   double rsi[];
   if (rsiHandle==INVALID_HANDLE||CopyBuffer(rsiHandle,0,0,1,rsi)!=1||rsi[0]>RSI_Threshold) return;
   
   double sum0=0,sumSq0=0;
   for(int i=0;i<BBPeriod_Bandwidth;i++){sum0+=sma[i];sumSq0+=sma[i]*sma[i];}
   
   double mean0=sum0/BBPeriod_Bandwidth;
   double std0=MathSqrt((sumSq0-sum0*sum0/BBPeriod_Bandwidth)/BBPeriod_Bandwidth);
   double upper0=mean0+BBDeviation_Bandwidth*std0;
   double lower0=mean0-BBDeviation_Bandwidth*std0;
   double bandwidth0=(upper0-lower0)/mean0;
   double sum1=0,sumSq1=0;
  
   for(int i=1;i<=BBPeriod_Bandwidth;i++){sum1+=sma[i];sumSq1+=sma[i]*sma[i];}
   double mean1=sum1/BBPeriod_Bandwidth;
   double std1=MathSqrt((sumSq1-sum1*sum1/BBPeriod_Bandwidth)/BBPeriod_Bandwidth);
   double lower1=mean1-BBDeviation_Bandwidth*std1;
   double prevClose=iClose(sym,ExpectedTimeframe,1);
   double prevHigh=iHigh(sym,ExpectedTimeframe,1);
   double currClose=iClose(sym,ExpectedTimeframe,0);
  
   if(bandwidth0<MinBandwidthThreshold||prevClose>lower1||currClose<=prevHigh) return;
  
   double sl=NormalizeDouble(bid-ATRMultiplierSL*atr[0],_Digits);
   double tp=NormalizeDouble(bid+ATRMultiplierTP*atr[0],_Digits);
   double lot=CalculateLotSize(sym,sl,true);
   OpenTrade(sym,true,sl,tp,lot);
   state.lastTradeTime=TimeCurrent();
}

#endif // __StrategyBBBandwidth_mqh__
