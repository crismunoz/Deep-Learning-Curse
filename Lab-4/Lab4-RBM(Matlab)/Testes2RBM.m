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

%% Reconstruçao 
ind_extra=randi(size(data.testData,1),1,20);
noisyData=data.testData(ind_extra,:)+sqrt(0.02).*randn(size(data.testData(ind_extra,:)));
[reconstructedData]=rbm.reconstructData(noisyData,1)';
figure(1);
for i = 1:20
    subplot(4,5,i);k=ind_extra(i);
    imagesc(reshape(data.testData(k,:),28,28));colormap gray
end

figure(2);
for i = 1:20
    subplot(4,5,i);
    imagesc(reshape(noisyData(i,:),28,28));colormap gray
end

figure(3);
for i = 1:20
    subplot(4,5,i);
    imagesc(reshape(reconstructedData(:,i),28,28));colormap gray
end
