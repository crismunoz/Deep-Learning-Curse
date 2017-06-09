clear all; close all; clc
load('Mnist.mat')
hiddenSize = 256;
autoenc1 = trainAutoencoder(xTrain,hiddenSize,'MaxEpochs',400,...
           'L2WeightRegularization',0.004,...
           'SparsityRegularization',4,...
           'SparsityProportion',0.15,...
           'UseGPU',true);
xReconstructed1 = predict(autoenc1,xTrain);

figure;
for i = 1:20
    subplot(4,5,i);
    imagesc(reshape(xTrain(:,i),28,28));colormap gray
end

figure;
for i = 1:20
    subplot(4,5,i);
    imagesc(reshape(xReconstructed1(:,i),28,28));colormap gray
end
view(autoenc1)
return
%%
% Obtindo as caracteristicas do autoencoder
features1 = encode(autoenc1,xTrain);
% Treinamento do classificador
softnet = trainSoftmaxLayer(features1,yTrain,'LossFunction','crossentropy','MaxEpochs',500);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'MaxEpochs'             - 1000.
% 'LossFunction'          - 'mse', 'crossentropy' (default)
% 'ShowProgressWindow'    -  true.
% 'TrainingAlgorithm'     - 'trainscg' 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                  

%Stack the encoders and the softmax layer to form a deep network.
deepnet = stack(autoenc1,softnet);

%Treinamento da  deep network.

deepnet_f = train(deepnet,xTrain,yTrain,'UseGPU','yes');

% Matrix de Confusão
figure(1)
X_type = deepnet(xTest);
plotconfusion(yTest,X_type);
figure(2)
X_type_f = deepnet_f(xTest);
plotconfusion(yTest,X_type_f);
%Evaluate(deepnet_f)
%%
features1 = encode(autoenc1,xTrain);
% Treinamento o segundo autoencoder com as caracteristicas obtidas do
% primeiro autoencoder
hiddenSize = 50;
autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',0.002,...
    'MaxEpochs',300, ...
    'SparsityRegularization',4,...
    'SparsityProportion',0.1,...
    'ScaleData',false,...
        'UseGPU',true);
%Extrai as caracteristicas na camada escondida(autoencoder 2).
features2 = encode(autoenc2,features1);

% Treinamento do classificador
softnet = trainSoftmaxLayer(features2,yTrain,'LossFunction','crossentropy','MaxEpochs',1000);

%Stack the encoders and the softmax layer to form a deep network.
deepnet = stack(autoenc1,autoenc2,softnet);

%Train the deep network.
deepnet_f = train(deepnet,xTrain,yTrain,'UseGPU','yes');

% Matrix de Confusão
figure(1)
X_type = deepnet(xTest);
plotconfusion(yTest,X_type);
figure(2)
X_type_f = deepnet_f(xTest);
plotconfusion(yTest,X_type_f);
Evaluate(deepnet_f)

%reset(gpuDevice(1))