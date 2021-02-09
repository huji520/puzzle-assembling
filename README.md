# 3D-bio-hackathon
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Jigsaw.svg/1920px-Jigsaw.svg.png" width="250">
Authors:

Omer Dan, Rafael Horowitz, Shahar Jacob, Eden Elmaliah


to train NN:
 
 before run model run command:
 
 module load tensorflow/2.0.0
 
 argumnets:
 
 ----------------------------------------------------------------------------------
 Long Argument    Short Argument    Default                choose:                                                  
 
--epochs                 -e           200                  number of epches 
--load                   -l           no load              load weights     
    
 
 ----------------------------------------------------------------------------------
 
 example 1:
 
   python3 NNGray.py -e 200 -l 200
 
 example 2:
  
   python3 NNGray.py
