function fixedDepthNets()
% Demo function for matrix neural networks
% For details, see our paper "Matrix Neural Networks" by Junbin Gao, Yi Guo, Zhiyong Wang at https://arxiv.org/abs/1601.03805.
%
% It implements the updates proposed in "Symmetry-invariant optimization in deep networks", arXiv:1511.01754, 2015.
%
% Version history: ver 0.5
% Maintainer: Yi Guo, Junbin Gao
% Date: Oct 2017
%
% NOTES:
% This uses MNIST data.
% Fixed network structure. See architecture parameters section below.

clear all
rand('seed', 1e6);
randn('seed',1e6);

% Load the MNIST dataset
Train_x = loadMNISTImages('./data/train-images.idx3-ubyte');
Train_x = reshape(Train_x, [28 28 size(Train_x,2)]);
labels = loadMNISTLabels('./data/train-labels.idx1-ubyte');
Train_y = zeros(length(labels),10);
for i=1:length(labels)
    Train_y(i, labels(i)+1) = 1;
end
clear labels
%% Load the MNIST test dataset
Test_x = loadMNISTImages('./data/t10k-images-idx3-ubyte');
Test_x = reshape(Test_x, [28 28 size(Test_x,2)]);
labels_all = loadMNISTLabels('./data/t10k-labels-idx1-ubyte');
Test_y = zeros(length(labels_all),10);
for i=1:length(labels_all)
    Test_y(i,labels_all(i)+1) = 1;
end
clear labels_all

%% Set up the architecture parameters
layerSize{1}.I = 28; layerSize{1}.J = 28; % input layer
layerSize{2}.I = 20; layerSize{2}.J = 20; % hidden layer
layerSize{3}.I = 16; layerSize{3}.J = 16; % hidden layer
layerSize{4} = 10; % hidden layer

matrix_net.layerSize = layerSize;
matrix_net.pool = 1;
matrix_net.classes = layerSize{end};

%%
% Pick the architecture. Arch1 of the paper is the Basic architecture and
% Arch2 is Arch1 + batch normalization.
type = {'Arch1'}; % {'Arch1','Arch2'};

% SGD updates
% B-SGD is the standard Euclidean update.
% SM is the update with the proposed scaled metric.
% UN is the update by constraining the filters to be on the unit-norm
% manifold.
method = {'B-SGD'}; %{'B-SGD','SM','UN'};

% Depth of the network
netDepth = length(layerSize) - 1;  % [2 4];


% Learning rate decay protocols
protocol = {'Bold_driver'}; % {'Bold_driver','Exp_decay'};


%% Options
mbatch = 100; % Batchsize
shuffle = true; % Random or sequential minibatches
runs = 1; % 10; % Number of runs for a particular choice of Arch and update scheme.
minEpochs = 25; % Minimum number of epochs before we start checking convergence.
numEpochs = 60; % Maximum number of epochs for trainining.
numValidEpochs = 50; % Number of epochs to search for the initial learning rate parameter.


%% Training

% Train over different learning rate protocols
for p = 1 : numel(protocol)
    
    % Train for different depths of the network
    for d = 1 : numel(netDepth)
        
        matrix_net.numLayers = netDepth(d) + 1;
        
        % Train over the architectures
        for t = 1 : numel(type)
            
            % Train over different updates: Euclidean, SM, and UN
            for m = 1 : numel(method)
                
                % Train the network - SGD - multiple runs
                for rr = 1 : runs
                    
                    % Perform grid search for finding the base learning rate
                    display(['Performing grid search for learning rate: Protocol ' protocol{p} ' type ' type{t} ' method ' method{m} ' run ' num2str(rr) ' depth = ' num2str(netDepth(d)) ]);
                    
                    % Create train and validation sets
                    ind = randperm(size(Train_x,3));  %,size(Train_x,1));
                    indTrain = ind(1:1000);
                    indValid = ind(end-499:end);
                    
                    gval_x = Train_x(:, :, indValid);
                    gval_y = Train_y(indValid,:);
                    gtrain_x = Train_x(:,:,indTrain);
                    gtrain_y = Train_y(indTrain,:);
                    gtest_x  = gval_x;
                    gtest_y  = gval_y;
                    gnumBatches = ceil(size(gtrain_x,3)/mbatch);
                    eta = 0:.5:5;
                    lambda = 10.^(-eta); % We search the base learning rate over {1e-2 to 1e-6}
                    trajmin = zeros(numel(lambda),1);
                    
                    
                    for kk = 1:numel(lambda)% Search over lambda choices
                        display([' lambda = ' num2str(lambda(kk))]);
                        gW = initializeMatrixWeights(matrix_net);
                        for i = 1:numValidEpochs
                            [gWoptim,infos] = trainMatrixNetwork(gW,gtrain_x,gtrain_y,gtest_x,gtest_y,gval_x,gval_y,matrix_net.pool,lambda(kk),gnumBatches,mbatch,shuffle,matrix_net.layerSize,method{m},type{t});
                            gW = gWoptim;
                            fprintf('Protocol = %s, type = %s, method = %s, run = %d, depth = %d, Epoch = %d, Cost = %3f, Train error = %3f, Validation error = %3f\n',protocol{p},type{t},method{m},rr,netDepth(d),i,infos.optimcost,infos.trainerror,infos.testerror);
                        end
                        trajmin(kk) = infos.validerror;
                        %trajmin(kk) = 0.01;  %infos.validerror;
                        
                        display(['Validation error for lambda = ' num2str(lambda(kk)) ' = ' num2str(trajmin(kk))]);
                    end
                    
                    
                    % Find lowest cost and corresponding lambda
                    [tttt, I] = min(trajmin);
                    lambdaOptim = lambda(I);
                    lambdaStart = lambdaOptim; % The base learning rate.
                    
                    display(['Protocol ' protocol{p} ' type ' type{t} ' method ' method{m} ' run ' num2str(rr) ' depth = ' num2str(netDepth(d)) ' Base optimal lambda ' num2str(lambdaOptim)]);
                    
                    % Create train, validation, and test sets
                    ind = randperm(size(Train_x,3));  %,size(Train_x,1));
                    indTrain = ind(1:50000);
                    indValid = ind(50001:end);
                    val_x = Train_x(:,:,indValid);
                    val_y = Train_y(indValid,:);
                    train_x = Train_x(:,:,indTrain);
                    train_y = Train_y(indTrain,:);
                    test_x  = Test_x;
                    test_y  = Test_y;
                    numBatches = ceil(size(train_x,3)/mbatch);
                    
                    optimcost = [];
                    trainerr = [];
                    testerr = [];
                    validerr = [];
                    
                    % Random initializations
                    W = initializeMatrixWeights(matrix_net);
                    
                    for i = 1 : numEpochs
                        [Woptim,infos] = trainMatrixNetwork(W,train_x,train_y,test_x,test_y,val_x,val_y,matrix_net.pool,lambdaOptim,numBatches,mbatch,shuffle,matrix_net.layerSize,method{m},type{t});
                        
                        %% Update information
                        W = Woptim;
                        optimcost = [optimcost; infos.optimcost];
                        %optimcost = [optimcost; rand(1)];
                        trainerr = [trainerr; infos.trainerror];
                        %trainerr = [trainerr; rand(1)];
                        testerr = [testerr; infos.testerror];
                        %testerr = [testerr; rand(1)];
                        validerr = [validerr; infos.validerror];
                        %validerr = [validerr; rand(1)];
                        fprintf('Protocol = %s, type = %s, method = %s, run = %d, depth = %d, Epoch = %d, Learning rate = %3f, Cost = %3f, Train error = %3f, Test error = %3f, Validation error = %3f\n', protocol{p}, type{t}, method{m}, rr, netDepth(d), i, lambdaOptim, infos.optimcost, infos.trainerror, infos.testerror, infos.validerror);
                        
                        % Update the learning rate
                        if strcmp(protocol{p},'Bold_driver')
                            
                            % Bold driver Protocol
                            if i >= 2 % Learning rate decay
                                if ((optimcost(end) > optimcost(end-1)) || isnan(optimcost(end)))
                                    lambdaOptim = lambdaOptim/2;
                                else
                                    lambdaOptim = 1.1*lambdaOptim;
                                end
                            end
                            
                        elseif strcmp(protocol{p},'Exp_decay')
                            
                            % Exponential decay protocol
                            if i >= 1 % Learning rate decay
                                lambdaOptim = 0.95*lambdaOptim;
                            end
                            
                        end
                        
                        % Stopping criterion
                        if strcmp(protocol{p},'Bold_driver')
                            if i > minEpochs
                                if trainerr(end) < 1e-5
                                    break;
                                end
                                
                                if (mod(i,5) == 0)
                                    if (validerr(end-4) < validerr(end))
                                        display(['Stopping Learning of Layer ' num2str(matrix_net.numLayers)]);
                                        break;
                                    end
                                end
                                
                                if (abs(validerr(end) - validerr(end-1)) < 1e-5)
                                    break;
                                end
                            end
                        end
                        
                    end % Over the number of epochs
                    
                    display(['Optimal cost = ' num2str(mean(optimcost(end)))]);
                    display(['Optimal training error = ' num2str(mean(trainerr(end)))]);
                    display(['Optimal test error = ' num2str(mean(testerr(end)))]);
                    display(['Optimal validation error = ' num2str(mean(validerr(end)))]);
                    
                    stats.optimcost = optimcost;
                    stats.trainerr = trainerr;
                    stats.testerr = testerr;
                    stats.validerr = validerr;
                    stats.lambdaOptim = lambdaStart;
                    %save(['./results/' method{m} '/' type{t} '_depth' num2str(matrix_net.numLayers) '_run' num2str(rr) '_' protocol{p} '_fixedEpochs' '.mat'],'stats');
                    
                end % Over the runs
                
            end % Over the SGD updates
            
        end % Over the architecture types
        
    end % Over the network depths
    
end % Over the protocols

