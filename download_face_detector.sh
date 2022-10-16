#!/bin/bash

LINK="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
FILE="face_detection/haarcascade_frontalface_default.xml"

curl -L $LINK --output $FILE

exit 1