#!/bin/bash

mkdir -p datasets/Exposure
gdown 1h96PnIuzyZLVAgi0SbTNlBg3aR8oAiuR
tar -xzf exposure-video.zip --directory datasets/Exposure
chmod -R 755 datasets/Exposure