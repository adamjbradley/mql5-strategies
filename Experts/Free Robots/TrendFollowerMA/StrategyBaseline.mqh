//=== StrategyBaseline.mqh ===
#ifndef __StrategyBaseline_mqh__
#define __StrategyBaseline_mqh__

// Random buys
//Uses ATR-based trailing stop for exits

// Externals assumed: DCPeriod, VolMaPeriod, ReentryCooldownBars, ExpectedTimeframe

//+------------------------------------------------------------------+
//| Core strategy evaluation (called every tick)                    |
//+------------------------------------------------------------------+
void Evaluate_Baseline(string sym, SymbolState* state)
{
   // make sure we have enough history for everything:
   int   neededBars = MathMax(
                    MathMax( SR_LookbackBars + SR_PivotRight,
                             DCPeriod ),
                    MathMax( VolMaPeriod + 1,
                             ATRPeriod  )
                  );
   if(Bars(sym,ExpectedTimeframe) < neededBars)
      return; // not warmed up yet

   // 1) enforce M5 chart
   if(ExpectedTimeframe!=PERIOD_M5 || _Period!=PERIOD_M5)
      return;

   // 2) sync check
   if(!SeriesInfoInteger(sym, ExpectedTimeframe, SERIES_SYNCHRONIZED))
      return;

   // 3) new-bar guard: only run entry logic once per M5 bar
   datetime curBar = iTime(sym, PERIOD_H1, 0);
   if(curBar==lastM5BarTime)
     return;
   lastM5BarTime = curBar;

   bool isLong = true;

   // 9) Calculate SL with minimum distance check
   double entryPrice = isLong ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
   double sl = 0.0;
   
   // Get minimum stop level in points
   long stopLevel = SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   double minStopDistPoints = stopLevel > 0 ? stopLevel : 10; // Default to 10 points if not specified
   
   // Convert to price
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   double minStopDist = minStopDistPoints * point;
   
   // Ensure SL is at valid distance
   if(isLong && (entryPrice - sl < minStopDist))
      sl = entryPrice - minStopDist;
   else if(!isLong && (sl - entryPrice < minStopDist))
      sl = entryPrice + minStopDist;
   
   // Calculate lot size based on adjusted SL
   double lot = CalculateLotSize(sym, sl, isLong);
   
   PrintFormat("%s: OpenTrade(isLong=%s, SL=%.5f, Lot=%.2f)", sym, isLong?"Y":"N", sl, lot);
   OpenTrade(sym, isLong, 0.0, 0.0, lot);

   // 10) record for cooldown
   state.lastTradeTime = TimeCurrent();
}

//+------------------------------------------------------------------+
#endif // __StrategyBaseline_mqh__
