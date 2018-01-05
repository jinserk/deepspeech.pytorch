.PHONY: all clean

IGNORE := $(shell bash -c "source ./path.sh; env | grep KALDI_ROOT | sed 's/^KALDI_ROOT=/export KALDI_ROOT:=/' > ./mkenv")
include ./mkenv
include $(KALDI_ROOT)/src/kaldi.mk

# kaldi subdir definition : DO NOT change the order!
KALDI_SUBDIRS = sgmm2 nnet2 nnet3 chain ivector online2 \
				gmm feat base matrix util tree transform \
                lat fstext hmm lm decoder kws cudamatrix nnet

KALDI_LDFLAGS = $(foreach d, $(KALDI_SUBDIRS), -L$(KALDI_ROOT)/src/$(d))
KALDI_LIBS = $(foreach d, $(KALDI_SUBDIRS), $(KALDI_ROOT)/src/$(d)/libkaldi-$(d).so)

CC = gcc
CXX = g++

# put a sharp sign before -Wextra if you're compiling 
# with an older gcc
CXXFLAGS += -fPIC -Wall -g -I$(KALDI_ROOT)/src
LDFLAGS += -shared $(KALDI_LDFLAGS)
LDLIBS += -Wl,-Bdynamic $(KALDI_LIBS) -lstdc++

TARGET = liblatgen.so

SRCS = latgen.cc
OBJS = $(SRCS:.cc=.o)
DEPS = $(SRCS:.cc=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(DEPS):%.d:%.cc
	$(CXX) $(CXXFLAGS) -MM $< >$@

include $(DEPS)

clean:
	$(RM) ./mkenv $(TARGET) $(OBJS) $(DEPS)

