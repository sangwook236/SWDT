/****************************************************************************
*
* Error.h
*
* Purpose:  Error checking routines.
*
* Author: Donovan Parks, July 2007
*
******************************************************************************/

#ifndef ERROR_H
#define ERROR_H

bool Error(const char* msg, const char* code, int data);

bool TraceInit(const char* filename);
void Trace(const char* msg);
void TraceClose();

#endif
