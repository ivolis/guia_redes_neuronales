
% Saco también E y V_o para testear la red       
function [F,E,V_o] = fitness(w, X, Nneu, size_inputs, hidden_layer_amount, size_outputs, V_d)
    for mu=randperm(length(X)) % recorro aleatoriamente todos los patrones      
        ind = 1;
        W_ks = reshape(w(ind:Nneu*size_inputs),[Nneu,size_inputs]);
        ind = ind + Nneu*size_inputs;
        b_k = reshape(w(ind:(ind+Nneu)-1),[Nneu,1]);
        ind = ind + Nneu;

        h_k = (W_ks*X(mu,:)') + b_k;
        V = tanh(h_k);


        for hid_layer=1:(hidden_layer_amount-1) % k -> j
            W_hid_layers = reshape(w(ind:(ind+Nneu*Nneu)-1),[Nneu,Nneu]);
            ind = ind + Nneu*Nneu;
            b_hid_layers = reshape(w(ind:(ind+Nneu*1)-1),[Nneu,1]);
            ind = ind + Nneu*1;

            h_hid_layer = (W_hid_layers*V) + b_hid_layers;
            V = tanh(h_hid_layer);
        end

        W_oj = reshape(w(ind:(ind+size_outputs*Nneu)-1),[size_outputs,Nneu]);
        ind = ind + Nneu*size_outputs;
        b_o = reshape(w(ind:(ind+1)-1),[size_outputs,1]);
        ind = ind + size_outputs;

        h_o = (W_oj*V) + b_o;
        V_o(mu,1) = tanh(h_o);
    end
    
    E=(1/length(X))*sum( (V_d-V_o).^2 ); % 1/Np
    F = 1 - E/4;
end