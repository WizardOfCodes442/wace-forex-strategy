//+------------------------------------------------------------------+
//|                                                     Research.mq5 |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
#include "..\RL\FQF.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 100000;
bool                 TrainMode = true;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Actor;
CNet                 Critic;
CFQF                 Scheduler;
int                  Models = 1;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State1;
CBufferFloat         Result;
vector<float>        ActorResult;
vector<float>        CriticResult;
vector<float>        SchedulerResult;
bool                 Sample = true;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
//--- load models
   float temp;
   if(!Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true) ||
      !Critic.Load(FileName + "Crt.nnw", temp, temp, temp, dtStudied, true) ||
      !Scheduler.Load(FileName + "Sch.nnw", dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *critic = new CArrayObj();
      CArrayObj *schedule = new CArrayObj();
      if(!CreateDescriptions(actor, critic, schedule))
        {
         delete actor;
         delete critic;
         delete schedule;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !Critic.Create(critic) || !Scheduler.Create(schedule))
        {
         delete actor;
         delete critic;
         delete schedule;
         return INIT_FAILED;
        }
      delete actor;
      delete critic;
      delete schedule;
     }
   else
      Sample = false;
   Scheduler.getResults(SchedulerResult);
   Models = (int)SchedulerResult.Size();
   Actor.getResults(ActorResult);
   Scheduler.SetUpdateTarget(Iterations);
   if(ActorResult.Size() % Models != 0)
     {
      PrintFormat("The scope of the scheduler does not match the scope of the Agent (%d <> %d)", Models, ActorResult.Size());
      return INIT_FAILED;
     }
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
   Critic.Save(FileName + "Crt.nnw", 0, 0, 0, TimeCurrent(), true);
   Scheduler.Save(FileName + "Sch.nnw", TimeCurrent(), true);
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int GetAction(CBufferFloat *results, int model, int total_models)
  {
   if(!results)
      return -1;
   int actions = results.Total() / total_models;
   vectorf temp;
   temp.Init(actions);
   int start = model * actions;
   for(int i = 0; i < actions; i++)
      temp[i] = results.At(start + i);
   temp = temp.CumSum()  / temp.Sum();
   int err_code;
   float random = (float)Math::MathRandomNormal(0.5, 0.4, err_code);
   if(random >= 0 && random <= 1)
      for(int i = 0; i < actions; i++)
         if(random <= temp[i])
            return i;
//---
   return (actions - 1);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SaveTotalBase(void)
  {
   int total = ArraySize(Buffer);
   if(total < 0)
      return true;
   int handle = FileOpen(FileName + ".bd", FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(handle < 0)
      return false;
   if(FileWriteInteger(handle, total) < INT_VALUE)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Save(handle))
        {
         FileClose(handle);
         return false;
        }
   FileFlush(handle);
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CreateDescriptions(CArrayObj *actor, CArrayObj *critic, CArrayObj *scheduler)
  {
//---
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
//---
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         return false;
     }
//---
   if(!scheduler)
     {
      scheduler = new CArrayObj();
      if(!scheduler)
         return false;
     }
//--- Actor
   actor.Clear();
   CLayerDescription *descr;
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (int)(HistoryBars * 12 + 9);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 5000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 300;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 100;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 50;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 1000;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 200;
   descr.window = 100;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 50;
   descr.window = 200;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 4;
   descr.window = 50;
   descr.step = 10;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 10
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = 4;
   descr.step = 10;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (int)(HistoryBars * 12 + 9);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 5000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 300;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 100;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 50;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 1000;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 200;
   descr.window = 100;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 50;
   descr.window = 200;
   descr.step = 10;
   descr.activation = TANH;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 4;
   descr.window = 50;
   descr.step = 10;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Scheduler
   scheduler.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (int)(HistoryBars * 12 + 9 + 40);
   descr.window = 0;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 5000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 300;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 100;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = 50;
   descr.window = 2;
   descr.step = 2;
   descr.window_out = 4;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 100;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFQF;
   descr.count = 10;
   descr.window_out = 32;
   descr.optimization = ADAM;
   if(!scheduler.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)(((double)MathRand() / 32767.0) * (total_tr - 1));
      int i = 0;
      i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      State1.AssignArray(Buffer[tr].States[i].state);
      if(IsStopped())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         return;
        }
      if(!Actor.feedForward(GetPointer(State1), 12, true) ||
         !Critic.feedForward(GetPointer(State1), 12, true))
         return;
      Actor.getResults(ActorResult);
      Critic.getResults(CriticResult);
      State1.AddArray(ActorResult);
      if(!Scheduler.feedForward(GetPointer(State1), 12, true))
         return;
      Scheduler.getResults(SchedulerResult);
      int agent = (Sample ? Scheduler.getSample() : Scheduler.getAction());
      if(agent < 0)
        {
         iter--;
         continue;
        }
      int actions = (int)(ActorResult.Size() / SchedulerResult.Size());
      float max_value = CriticResult[agent * actions];
      for(int j = 1; j < actions; j++)
         max_value = MathMax(max_value, CriticResult[agent * actions + j]);
      SchedulerResult[agent] = Buffer[tr].Revards[i];
      Result.AssignArray(SchedulerResult);
      //---
      if(!Scheduler.backProp(GetPointer(Result), 0.0f, NULL))
         return;
      int agent_action = agent * actions + Buffer[tr].Actions[i];
      CriticResult[agent_action] = Buffer[tr].Revards[i];
      Result.AssignArray(CriticResult);
      //---
      if(!Critic.backProp(GetPointer(Result)))
         return;
      ActorResult.Fill(0);
      ActorResult[agent_action] = Buffer[tr].Revards[i] - max_value;
      Result.AssignArray(ActorResult);
      //---
      if(!Actor.backProp(GetPointer(Result)))
         return;
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("Actor %.2f%% -> Error %.8f\n", iter * 100.0 / (double)(Iterations), Actor.getRecentAverageError());
         str += StringFormat("Critic %.2f%% -> Error %.8f\n", iter * 100.0 / (double)(Iterations), Critic.getRecentAverageError());
         str += StringFormat("Scheduler %.2f%% -> Error %.8f\n", iter * 100.0 / (double)(Iterations), Scheduler.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %10.7f", __FUNCTION__, __LINE__, Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %10.7f", __FUNCTION__, __LINE__, Critic.getRecentAverageError());
   PrintFormat("%s -> %d -> %10.7f", __FUNCTION__, __LINE__, Scheduler.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
