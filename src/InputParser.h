#ifndef INPUTPARSER_H_INCLUDED
#define INPUTPARSER_H_INCLUDED

#include "Header.h"

void readIndexParam(int , char**, IndexParam & );
void readQueryParam(int nargs, char** args, QueryParam &);

void loadtxtDatabase(int, char**, const int & , const int &, MatrixXf & );
void loadtxtQuery(int, char**, const int &, const int &, MatrixXf & );

#endif // INPUTPARSER_H_INCLUDED
