#!/bin/bash

predict.sh --cuda no --decoder greedy test/conan1-8k.wav
predict.sh --cuda no --decoder beam test/conan1-8k.wav

predict.sh --cuda no --decoder greedy test/conan2-8k.wav
predict.sh --cuda no --decoder beam test/conan2-8k.wav

predict.sh --cuda no --decoder greedy test/conan3-8k.wav
predict.sh --cuda no --decoder beam test/conan3-8k.wav
