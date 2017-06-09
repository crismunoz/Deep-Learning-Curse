close all; clc;clear all
%% Primeiro exemplo
% Data
[xTrain ,yTest]=iris_dataset;
% Treinamento
autoenc        = trainAutoencoder(xTrain,'UseGPU',true,'MaxEpochs',2000);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'EncoderTransferFunction' -  'logsig'(default) , 'satlin'
% 'DecoderTransferFunction' -  'logsig'(default) , 'satlin' , 'purelin' 
% 'MaxEpochs'               -   1000.
% 'L2WeightRegularization'  -   0.001.
% 'LossFunction'            -   'msesparse' (única).
% 'ShowProgressWindow'      -   'true'.
% 'SparsityProportion'      -   Must be between 0 and 1. (0.05).
% 'SparsityRegularization'  -   1.
% 'TrainingAlgorithm'       -   Only the value 'trainscg'
% 'UseGPU'                  -   false(default).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstrução de dados (todos ois dados de treinamento)
XReconstructed = predict(autoenc,xTrain);
% Calculo do erro de reconstrução
mseError       = mse(xTrain-XReconstructed)
return
%% Funções Basicas
% Encode
var_enc=encode(autoenc,xTrain(:,1))
% Decode
var_Enc_Dec=decode(autoenc,var_enc)
% Predição
predict(autoenc,xTrain(:,1))
% Vista da Rede
view(autoenc)
%% Segundo exemplo
autoenc2        = trainAutoencoder(xTrain,8,'MaxEpochs',500,'DecoderTransferFunction','satlin','UseGPU',true);
XReconstructed2 = predict(autoenc2,xTrain);
mseError2       = mse(xTrain-XReconstructed2)
view(autoenc2)
%% Terceiro exemplo (PCA)
autoenc3        = trainAutoencoder(xTrain,4,'MaxEpochs',2000,'DecoderTransferFunction','purelin','UseGPU',true);
XReconstructed3 = predict(autoenc3,xTrain);
mseError3       = mse(xTrain-XReconstructed3)
view(autoenc3)