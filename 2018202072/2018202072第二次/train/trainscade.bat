opencv_traincascade.exe -data xml -vec pos.vec -bg negdata.txt -numPos 24 -numNeg 16 -numStages 1 -featureType LBP-w 40 -h 40 -minHitRate 0.999 -maxFalseAlarmRate 0.2 -weightTrimRate 0.95
pause