//+------------------------------------------------------------------+
//|                                                   PipeClient.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#define PIPE_NAME "\\\\.\\pipe\\PriceFeedProvider.Server"
#include <Files\FilePipe.mqh>

CFilePipe  ExtPipe;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnInit()
  {
  
//--- wait for pipe server
   while(!IsStopped())
     {
    if(ExtPipe.Open(PIPE_NAME,FILE_READ|FILE_WRITE|FILE_BIN)!=INVALID_HANDLE) break;
      Sleep(250);
     }
   Print("Client: pipe opened");
//--- send welcome message
   if(!ExtPipe.WriteString(__FILE__+" on MQL5 build "+IntegerToString(__MQ5BUILD__)))
     {
      Print("Client: sending welcome message failed");
      return;
     }

  }
  
  
  void OnTick()
  {
//---


      ExtPipe.Flush();
      double price = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
      printf("Price: %.2f",price);
      if(!ExtPipe.WriteDouble(price))
     {
         Print("Client: sending price failed");
         return;
     }
     

  }
  
  
  void OnDeinit(const int reason)
  {
//---
     ExtPipe.Close();
  }
//+------------------------------------------------------------------+
