# Detect OS 
ifndef TARGETOS
  TARGETOS := $(shell uname -s)
endif

# Use vecLib/Accelerate on macOS
ifeq ($(TARGETOS), Darwin)
  VECLIB = T
endif

VINE = ${HOME}/vine

# Points to top-level directory of PHAST installation
PHAST = ${HOME}/phast

CC = gcc
LN = ln

LIB = ${VINE}/lib
INC = ${VINE}/include
BIN = ${VINE}/bin
TARGETLIB = ${LIB}/libvine.a

PHASTLIB = ${PHAST}/lib
PHASTINC = ${PHAST}/include

#for debugging
#CFLAGS = -g -fno-inline -Wall 
# for best performance
CFLAGS = -O3 -Wall
# for profiling
#CFLAGS = -O2 -g 

# for AddressSanitizer
#CFLAGS = -O1 -g -fsanitize=address -fno-omit-frame-pointer
#LDFLAGS = -fsanitize=address -fno-omit-frame-pointer

CFLAGS += -I${INC} -I${PHASTINC} -I${PHAST}/src/lib/pcre -fno-strict-aliasing
LIBPATH = -L${LIB} -L${PHASTLIB}

#LIBS = -lphast -llapack -ltmg -lblaswr -lc -lf2c -lm
# LAPACK / BLAS configuration (mirror PHAST behavior)

ifdef VECLIB
  # vecLib / Accelerate (macOS, same as PHAST)
  CFLAGS += -DVECLIB
  LIBS   = -lvine -lphast -framework Accelerate -lc -lm
else
  # Fallback: CLAPACK (only if you actually use CLAPACK on this machine)
  # You can tune CLAPACKPATH if needed, but this mirrors PHAST defaults.
  ifndef CLAPACKPATH
    CLAPACKPATH = /usr/local/software/clapack
  endif
  F2CPATH = ${CLAPACKPATH}/F2CLIBS

  CFLAGS += -I${CLAPACKPATH}/INCLUDE -I${F2CPATH}
  LIBS   = -lvine -lphast -llapack -ltmg -lblaswr -lc -lf2c -lm
  LIBPATH += -L${F2CPATH}
endif

