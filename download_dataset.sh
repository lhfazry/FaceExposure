#!/bin/bash

mkdir -p datasets/Exposure
gdown 1qQx85P-b0UAu4EJ9H7y3ukLFu6wtO_Ut
tar -xzf exposure-video.zip --directory datasets/Exposure
chmod -R 755 datasets/Exposure