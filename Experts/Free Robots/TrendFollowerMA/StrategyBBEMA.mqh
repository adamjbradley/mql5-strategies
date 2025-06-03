//=== StrategyBBEMA.mqh ===
#ifndef __StrategyBBEMA_mqh__
#define __StrategyBBEMA_mqh__

void EvaluateBB_EMA(string sym, SymbolState* state) {
   if (SeriesInfoInteger(sym, ExpectedTimeframe, SERIES_SYNCHRONIZED) == false) return;
   if (PositionSelect(sym)) { HandleExit(sym, state); return; }
   if (TimeCurrent() - state.lastTradeTime < ReentryCooldownBars * PeriodSeconds(_Period)) return;
   double bid, ask;
   if (!SymbolInfoDouble(sym, SYMBOL_BID, bid) || !SymbolInfoDouble(sym, SYMBOL_ASK, ask)) return;
   double ema[], atr[];
   if (CopyBuffer(state.emaHandle, 0, 0, BBPeriod, ema) != BBPeriod) return;
   if (CopyBuffer(state.atrHandle, 0, 0, 1, atr) != 1 || atr[0] <= 0) return;
   double sum=0, sumSq=0;
   for(int i=0;i<BBPeriod;i++){ sum+=ema[i]; sumSq+=ema[i]*ema[i]; }
   double mean=sum/BBPeriod;
   double stddev=MathSqrt((sumSq-sum*sum/BBPeriod)/BBPeriod);
   double upperBand=mean+BBDeviation*stddev;
   double lowerBand=mean-BBDeviation*stddev;
   bool isBuy=false; double sl=0, tp=0;
   if(bid>upperBand){ isBuy=true; sl=NormalizeDouble(bid-ATRMultiplierSL*atr[0],_Digits); tp=NormalizeDouble(bid+ATRMultiplierTP*atr[0],_Digits); }
   else if(bid<lowerBand){ isBuy=false; sl=NormalizeDouble(ask+ATRMultiplierSL*atr[0],_Digits); tp=NormalizeDouble(ask-ATRMultiplierTP*atr[0],_Digits); }
   else return;
   double lot=CalculateLotSize(sym,sl,isBuy);
   OpenTrade(sym,isBuy,sl,tp,lot,NULL);
   state.lastTradeTime=TimeCurrent();
}

#endif // __StrategyBBEMA_mqh__
