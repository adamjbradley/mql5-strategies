//+------------------------------------------------------------------+
//|                                                   PipeClient.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#define PIPE_NAME "\\\\.\\pipe\\PriceFeedConsumer.Server"
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



      double price = 0;
      
      // Send read request (for simplification it is defined to be 1)
      if(!ExtPipe.WriteInteger(1)) return;
      
      if(!ExtPipe.ReadDouble(price))
     {
         Print("Client: sending price failed");
         return;
     }
     
     printf("Price: %.2f",price);
     

  }
  
  
  void OnDeinit(const int reason)
  {
//---
     ExtPipe.Close();
  }
//+------------------------------------------------------------------+
