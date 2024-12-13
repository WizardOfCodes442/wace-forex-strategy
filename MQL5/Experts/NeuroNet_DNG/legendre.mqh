//+------------------------------------------------------------------+
//|                                                Legendre Wavelets |
//|                                              Copyright 2024, DNG |
//|                                 http://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#define Legendre4(x)    Legendre(4,2*x-1)
#define Legendre6(x)    Legendre(6,2*x-1)
#define Legendre8(x)    Legendre(8,2*x-1)
#define Legendre10(x)   Legendre(10,2*x-1)
#define Legendre12(x)   Legendre(12,2*x-1)
#define Legendre14(x)   Legendre(14,2*x-1)
#define Legendre16(x)   Legendre(16,2*x-1)
#define Legendre18(x)   Legendre(18,2*x-1)
#define Legendre20(x)   Legendre(20,2*x-1)
//+------------------------------------------------------------------+
//| Legendre Polynoms                                                |
//+------------------------------------------------------------------+
double Legendre(uint poly, double x)
  {
   double result = 0;
   switch(poly)
     {
      case 0:
         result = 1.0;
         break;
      case 1:
         result = x;
         break;
      default:
         result = ((2.0 * (poly - 1.0) + 1.0) * x * Legendre(poly - 1, x) - (poly - 1) * Legendre(poly - 2, x)) / poly ;
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
