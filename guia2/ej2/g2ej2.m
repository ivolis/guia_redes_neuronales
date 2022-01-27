%% XOR de 2 y 4 entradas
clc;clear all;

% XOR de 4 entradas =======================================================
eta_XOR_4 = 0.01;
X_XOR_4 = 2*[0,0,0,0;
     0,0,0,1;
     0,0,1,0;
     0,0,1,1;
     0,1,0,0;
     0,1,0,1
     0,1,1,0
     0,1,1,1
     1,0,0,0
     1,0,0,1
     1,0,1,0
     1,0,1,1
     1,1,0,0
     1,1,0,1
     1,1,1,0
     1,1,1,1]-1;
V_d_XOR_4 = 2*[0;1;1;0;1;0;0;1;1;0;0;1;0;1;1;0]-1;
Nneu_XOR_4=10;
E_target_XOR_4=1e-4;

% XOR de 2 entradas =======================================================
eta_XOR_2 = 0.015;
X_XOR_2 = [-1 -1;1 -1;-1 1;1 1];
V_d_XOR_2 = [-1;1;1;-1]; 
Nneu_XOR_2=4;
E_target_XOR_2=1e-9;
% =========================================================================

% DEFINO CUANTAS ENTRADAS VOY A USAR ======================================
eta=eta_XOR_4;
X=X_XOR_4; % entradas
V_d=V_d_XOR_4; %salidas deseadas
Nneu=Nneu_XOR_4;
E_target=E_target_XOR_4;
% =========================================================================

% 4 capas:
% 3 hidden de Nneu neuronas (m=1,2,3) -> V_k , V_j y V_i
% y la capa de salida con 1 sola neurona (m=4) -> V_o (out)
% y una entrada de 2 o 4 valores (m=0). -> V_s (start)

% Defino los pesos sinapticos iniciales
media=0;
varianza=0.5;

%-> normrnd(media,varianza,fila,column)
W_ks = normrnd(media,varianza,Nneu,length(X(1,:)));
W_jk = normrnd(media,varianza,Nneu,Nneu); 
W_ij = normrnd(media,varianza,Nneu,Nneu);
W_oi = normrnd(media,varianza,1,Nneu);

W_ks_inicial = W_ks;
W_jk_inicial = W_jk;
W_ij_inicial = W_ij;
W_oi_inicial = W_oi;
% Guardo los iniciales para ver que onda la CV

% Defino los bias
b_k = normrnd(media,varianza,Nneu,1);
b_j = normrnd(media,varianza,Nneu,1);
b_i = normrnd(media,varianza,Nneu,1);
b_o = normrnd(media,varianza,1,1);

% Inicializo los deltas y los bias en 0
[dW_oi , dW_ij , dW_jk , dW_ks] = deal(0);
[db_o , db_i , db_j , db_k] = deal(0);


% Aprendizaje
epoca=0; 
max_epoca=250; % Si no encuentra una cvg, lo paro yo.
E=1; % Para que entre al while
V_o=zeros(length(V_d),1); % Inicializo las salidas reales en 0.
while(E>E_target)
    E=0;
    
    %Quiero evitar un loop infinito 
    if (epoca==max_epoca)
        break
    end
    epoca=epoca+1;
    
    % Forwarding y Back
    for mu=randperm(length(X)) % recorro aleatoriamente todos los patrones
        % Entrada a capa k
        h_k = (W_ks*X(mu,:)') + b_k;
        V_k = tanh(h_k);
        
        % Capa k a capa j
        h_j = (W_jk*V_k) + b_j;
        V_j = tanh(h_j);
        
        % Capa j a capa i
        h_i = (W_ij*V_j) + b_i;
        V_i = tanh(h_i);
        
        % Capa i a capa o
        h_o = (W_oi*V_i) + b_o;
        V_o(mu) = tanh(h_o); % Guardo salidas reales para c/patron
        
        %==================================================================
        % Backpropagation: aprovecho que tengo los h/V para cada capa y voy
        % ya corriendo el algoritmo.
        %==================================================================
        
        % Actualizo pesos oi y bias o
        delta_o = (1-tanh(h_o).^2)*(V_d(mu)-V_o(mu));
        dW_oi = dW_oi + eta*delta_o*V_i'; % w_12=eta+v_2*d_1, por eso asi
        db_o = db_o + eta*delta_o;
        
        % Actualizo pesos ij y bias i
        delta_i = (1-tanh(h_i).^2).*(W_oi'*delta_o);
        dW_ij = dW_ij + eta*delta_i*V_j';
        db_i = db_i + eta*delta_i;
        
        % Actualizo pesos jk y bias j
        delta_j=(1-tanh(h_j).^2).*(W_ij'*delta_i);
        dW_jk = dW_jk + eta*delta_j*V_k';
        db_j = db_j + eta*delta_j;
        
        % Actualizo pesos ks y bias k
        delta_k=(1-tanh(h_k).^2).*(W_jk'*delta_j);
        dW_ks = dW_ks + eta*delta_k*X(mu,:);
        db_k = db_k + eta*delta_k;  
   
        
        % Uso .* porque 1-tanh(h) puede devolver un vector y es para
        % multiplicar elemento con elemento => [2;2] .* [2;3] = [4;6]
    end
    
    E=0.5*sum( (V_d-V_o).^2 ); % .^2 para q' eleve c/elemento
    errores(epoca)=E; % Guardo para graficar
    errores(end)
    
    if(E>0) % Actualizo solo si el error sigue siendo >0 
        W_ks = W_ks + dW_ks;
        W_jk = W_jk + dW_jk;
        W_ij = W_ij + dW_ij;
        W_oi = W_oi + dW_oi;
        
        b_o = b_o + db_o;
        b_i = b_i + db_i;
        b_j = b_j + db_j;
        b_k = b_k + db_k;
    end
    % La condición del while es para que siga corriendo post-actualización.
end

%epocas=linspace(1,epoca,epoca);

figure(1)
plot(errores)
xlabel('Época')
ylabel('Error')
%plot(epocas,errores)
hold on
grid on
