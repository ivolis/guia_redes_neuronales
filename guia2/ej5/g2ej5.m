%% XOR de 2
clc;clear all;

% XOR de 2 entradas =======================================================
X_XOR_2 = [-1 -1;1 -1;-1 1;1 1];% entradas
V_d_XOR_2 = [-1;1;1;-1]; %salidas deseadas
Nneu_XOR_2=4;

% =========================================================================

% DEFINO LAS VARIABLES ====================================================
X=X_XOR_2;
V_d=V_d_XOR_2;
Nneu=Nneu_XOR_2;
% =========================================================================

% Defino los pesos sinapticos iniciales
media=0;
varianza=0.3;

W_ks = normrnd(media,varianza,Nneu,length(X(1,:)));
W_jk = normrnd(media,varianza,Nneu,Nneu); 
W_oj = normrnd(media,varianza,1,Nneu);

% Defino los bias
b_k = normrnd(media,varianza,Nneu,1);
b_j = normrnd(media,varianza,Nneu,1);
b_o = normrnd(media,varianza,1,1);

% Inicializo los deltas y los bias en 0
[dW_oj , dW_jk , dW_ks] = deal(0);
[db_o , db_j , db_k] = deal(0);


% Aprendizaje
V_o=zeros(length(V_d),1);
varianza_vecinos = 0.5; % para la eleccion de vecinos
E_target = 1e-3;
T_inicial = 20; % Arranco en T altas
T_corte = 150e-6; % enfrio hasta una T final
alfa=0.997%0.9997;
beta=1; % cte boltzmann

            
E_1=1; % Para entrar al while
T=T_inicial;
iter=0;
while(E_1>E_target)
    iter=iter+1;
    % Actual
    for mu=randperm(length(X)) % recorro aleatoriamente todos los patrones
        % Entrada a capa k
        h_k = (W_ks*X(mu,:)') + b_k;
        V_k = tanh(h_k);

        % Capa k a capa j
        h_j = (W_jk*V_k) + b_j;
        V_j = tanh(h_j);

        % Capa j a capa o
        h_o = (W_oj*V_j) + b_o;
        V_o(mu) = tanh(h_o); % Guardo salidas reales para c/patron 
    end

    E_1 = 1/2*sum((V_d - V_o).^2); % E_1 == E error actual
    
    % Vecinos
    delta_Wks = normrnd(0,varianza_vecinos,Nneu,length(X(1,:)));
    delta_Wjk = normrnd(0,varianza_vecinos,Nneu,Nneu);
    delta_Woj = normrnd(0,varianza_vecinos,1,Nneu);
    
    delta_bk = normrnd(media,varianza,Nneu,1);
    delta_bj = normrnd(media,varianza,Nneu,1);
    delta_bo = normrnd(media,varianza,1,1);
    
    W_ks = W_ks + delta_Wks;
    W_jk = W_jk + delta_Wjk;
    W_oj = W_oj + delta_Woj;
    
    b_k = b_k + delta_bk;
    b_j = b_j + delta_bj;
    b_o = b_o + delta_bo;
    
    
    for mu=randperm(length(X)) % recorro aleatoriamente todos los patrones
        % Entrada a capa k
        h_k = (W_ks*X(mu,:)') + b_k;
        V_k = tanh(h_k);

        % Capa k a capa j
        h_j = (W_jk*V_k) + b_j;
        V_j = tanh(h_j);

        % Capa j a capa o
        h_o = (W_oj*V_j) + b_o;
        V_o(mu) = tanh(h_o); % Guardo salidas reales para c/patron 
    end    
    
    E_2 = 1/2*sum((V_d - V_o).^2); % E_2 == E* error vecino
    
    deltaE=E_2-E_1;
    if deltaE >= 0
        if (binornd(1,exp(-deltaE/(beta*T)))==0) % no acepto
            W_ks = W_ks - delta_Wks;
            W_jk = W_jk - delta_Wjk;
            W_oj = W_oj - delta_Woj;
            b_k = b_k - delta_bk;
            b_j = b_j - delta_bj;
            b_o = b_o - delta_bo;
        else % acepto
            E_1=E_2;
        end
    else % acepto
        E_1=E_2;
    end
    % Enfriamiento
    T=T*alfa;
    if(T<T_corte)
        disp("no convergio")
        % llego a un T bajisima (ya ni va a aceptar cambios) por lo que no
        % tiene sentido seguir bajando la temperatura...
        break
    end
    E_1
    errores(iter)=E_1;
end

figure(1)
plot(errores)
title(['Error final con Simulated Annealing = ' num2str(errores(end))])
xlabel('Iteraciones')
ylabel('Error actual')
hold on
grid on
