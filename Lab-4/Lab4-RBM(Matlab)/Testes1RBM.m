clear all; close all; clc
addpath('DeeBNet');more off;
% Restricted Boltzmann Machine
%% Carrega Formato da base de dados
data=DataClasses.DataStore();
%load('Mnist.mat')
load('Mnist_small.mat')
data.valueType=ValueType.binary;
data.trainData=xTrain';
data.trainLabels=yTrain';
data.testData=xTest';
data.testLabels=yTest';
data.normalize('minmax');
%% Criando uma RBM como autoencoder
hiddenSize=256;
rbmParams=RbmParameters(hiddenSize,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.gpu=1;
rbmParams.maxEpoch=200;
rbm=GenerativeRBM(rbmParams);
%% Treinamento
rbm.train(data);
%% Obter caracteristicas
ind_extra=randi(size(data.testData,1),1,20);
[extractedFeature]=rbm.getFeature(data.testData(ind_extra,:),1);
% Generate nova data
[generatedData]=rbm.generateData(extractedFeature)';

% Plot Resultados
figure(1);
for i = 1:20
    subplot(4,5,i);k=ind_extra(i);
    imagesc(reshape(data.testData(k,:),28,28));colormap gray
end

figure(2);
for i = 1:20
    subplot(4,5,i);
    imagesc(reshape(generatedData(:,i),28,28));colormap gray
end
return