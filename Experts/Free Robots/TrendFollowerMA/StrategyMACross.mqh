//=== StrategyMACross.mqh ===
#ifndef __StrategyMACross_mqh__
#define __StrategyMACross_mqh__

void EvaluateMA_Cross(string sym, SymbolState* state) {
   if (SeriesInfoInteger(sym,ExpectedTimeframe,SERIES_SYNCHRONIZED)==false) return;
   if (PositionSelect(sym)) { HandleExit(sym,state); return; }
   if (TimeCurrent()-state.lastTradeTime<ReentryCooldownBars*PeriodSeconds(_Period)) return;
   double bid, ask;
   if (!SymbolInfoDouble(sym,SYMBOL_BID,bid)||!SymbolInfoDouble(sym,SYMBOL_ASK,ask)) return;
   double fastMA[], slowMA[], atr[];
   if (CopyBuffer(state.fastMAHandle,0,0,1,fastMA)!=1||CopyBuffer(state.slowMAHandle,0,0,1,slowMA)!=1) return;
   if (CopyBuffer(state.atrHandle,0,0,1,atr)!=1||atr[0]<=0) return;
   bool isBuy=false; double sl=0,tp=0;
   if(fastMA[0]>slowMA[0]){isBuy=true; sl=NormalizeDouble(bid-ATRMultiplierSL*atr[0],_Digits); tp=NormalizeDouble(bid+ATRMultiplierTP*atr[0],_Digits);}
   else if(fastMA[0]<slowMA[0]){isBuy=false; sl=NormalizeDouble(ask+ATRMultiplierSL*atr[0],_Digits); tp=NormalizeDouble(ask-ATRMultiplierTP*atr[0],_Digits);}
   else return;
   double lot=CalculateLotSize(sym,sl,isBuy);
   OpenTrade(sym,isBuy,sl,tp,lot, NULL);
   state.lastTradeTime=TimeCurrent();
}

#endif // __StrategyMACross_mqh__
