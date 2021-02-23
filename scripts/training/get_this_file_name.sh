#!/bin/sh

echo #==============================================#
echo # CDLEML Process
echo #==============================================#

ME=`basename "$0"`

echo my filename is: $ME

BASE=get_my_
MEPART=`echo $ME | sed 's/get_my_//' | sed 's/.sh//'`
echo $MEPART