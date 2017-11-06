#!/bin/bash

cuda="yes"

predict.sh --cuda $cuda --decoder greedy test/conan1-8k.wav
predict.sh --cuda $cuda --decoder beam test/conan1-8k.wav

predict.sh --cuda $cuda --decoder greedy test/conan2-8k.wav
predict.sh --cuda $cuda --decoder beam test/conan2-8k.wav

predict.sh --cuda $cuda --decoder greedy test/conan3-8k.wav
predict.sh --cuda $cuda --decoder beam test/conan3-8k.wav
