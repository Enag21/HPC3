############## FREETYPE SUPPORT / SOPORTE PARA FREETYPE#####################
#
# Uncomment the following line if you do not wish to compile PNGwriter
# with FreeType2 support.
# (Please note that if you compile PNGwriter without FreeType2 support,
# all projects that you compile that use PNGwriter must be compiled with
# -DNO_FREETYPE).
#
# Descomenta la siguiente linea si no quieres compilar PNGwriter
# con soporte para FreeType2.
# (Nota que si compilas PNGwriter sin soporte para FreeType2, todos los
# proyectos que usen PNGwriter que compiles deberan ser compilados con
# -DNO_FREETYPE).
#
#
P_FREETYPE = 1
#
# Alternatively, when compiling, call make with the option P_FREETYPE=1


PREFIX=/usr/local

DESTDIR=



ifdef P_FREETYPE
FT_ARG = -DNO_FREETYPE
else
FT_ARG = `freetype-config --cflags` `freetype-config --libs`
endif

# we need to use the system default compiler here, to be compatible
# with the system libpng, that was compiled with thatr version, too!
CXX=/usr/bin/g++

CXXFLAGS= -O3  #-Wall -Wno-deprecated $(FT_ARG)

INC=  -I../src/ -I$(PREFIX)/include/

LIBS= -L../src -L$(PREFIX)/lib/ -lz -lpng -lpngwriter

INSTALL=install

SELF=make.include.linux
