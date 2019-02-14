function [result] = feedForwardAE(W,b, data)

% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.

m = size(data, 2);
z_2 = W * data + repmat(b, 1, m);
a_2 = sigmoid(z_2);

activation = a_2;
% activation = activation.*(repmat(IndWeights,1,size(data,2)));

result = activation;

% [~,SortedInd] = sort(activation,1,'descend');
% Ind = SortedInd(1:numChoices,:);
% for ii = 1:size(Ind,2)
%     result(:,ii) = activation(Ind(:,ii),ii);
% end

%-------------------------------------------------------------------

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
