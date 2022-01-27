clc;clear all;

% XOR de 2 entradas =======================================================
X = [-1 -1;1 -1;-1 1;1 1];
V_d = [-1;1;1;-1]; 

size_inputs=length(X(1,:)); % cantidad de entradas
size_outputs=1; % cantidad de salidas
% =========================================================================


% Parametros de red =======================================================
Nneu=4; % neuronas por capa oculta
hidden_layer_amount=2;
media=0; % para los pesos sinapticos
varianza=0.5; % iniciales
%==========================================================================


% Parámetros del algoritmo ================================================
fitness_threshold = 1-1e-9/4;
pCross = 0.1% 0.03/0.1/0.2
crosses=0;
mu_mut=0;
var_mut=0.25;
N=101; % cantidad de individuos
%==========================================================================
% Aclaracion: Tiene que ser impar inicialmente para dejar lugar al
% individuo elite en cada generacion.
if mod(N,2)==0
   error('Error. La cantidad de inviduos de la poblacion (N) debe ser un numero impar')
end



% cantidad de pesos entre s y k y bias de k
n_wks_bk=Nneu*size_inputs+Nneu*1; 

% cantidad de pesos entre k y j (puede expandirse a mas capas ocultas)
% y bias de j (y/o bias de las otras)
n_w_hid_b_hid=(Nneu*Nneu+Nneu)*(hidden_layer_amount-1);

% cantidad de pesos entre k y j y bias de o
n_woj_bo=Nneu*size_outputs+size_outputs*1; 

for i=1:N % Creo poblacion (gen 1)
    W(:,i)=normrnd(media,varianza,n_wks_bk+n_w_hid_b_hid+n_woj_bo,1);
    F(i,1)=fitness(W(:,i),X,Nneu,size_inputs, hidden_layer_amount, size_outputs, V_d);
end
% Llamo W a la población de w's (pesos sinap y biases) de cada individuo.


[elite_fitness, elite_index] = max(F); % Elite de gen 1

iter=0;
while(elite_fitness < fitness_threshold) % Proximas generaciones
    
    iter=iter+1;
    
    p_rep=F/sum(F); % prob de repetir (de dejar descendencia?)
    
    
    % Voy armando la nueva gen n
    % El elite "pasa directo" a la prox gen (no hace crossover ni mutacion)
    W_new_gen(:,N) = W(:,elite_index);
     
    % Descendencia de N-1 indivs ( reproduccion )
    for i=1:N-1
    descendant_index = find(mnrnd(1,p_rep) == 1);
    W_new_gen(:,i) = W(:,descendant_index);
    end
    % Puede ser que el que tenía F_elite en gen 1 mute y haga cross para
    % llegar a gen 2, pero en gen 2 igual ya me aseguré ante de tener un
    % indiv de elite que no lo toqué.
    
    % Crossover
    rnd=randperm(N-1); % creo un vector de indices aleatorios
    for i=1:2:N-1 % :2 para agarrar "de a pares"
        if(binornd(1,pCross))
            crosses=crosses+1;
            % Selecciono un lugar de corte al azar entre los pesos de los
            % individuos seleccionados al azar
            ind_cross = unidrnd(size(W_new_gen,1));        
            W_newG_aux=W_new_gen; % Para no sobreescribir
            W_new_gen(ind_cross:end,rnd(i)) = W_newG_aux(ind_cross:end,rnd(i+1));
            W_new_gen(ind_cross:end,rnd(i + 1)) = W_newG_aux(ind_cross:end,rnd(i));
        end
    end
    
    % Mutacion
    r = normrnd(mu_mut, var_mut, size(W));
    W = W_new_gen + r;
    % elimino la mutación del ultimo index (es elite)
    W(:,N) = W(:,N) - r(:,N);
    
    
    for i=1:N % Saco el fitness de la poblacion n
        F(i,1)=fitness(W(:,i),X,Nneu,size_inputs, hidden_layer_amount, size_outputs, V_d);
    end
    [elite_fitness, elite_index] = max(F);
    fitness_evol(iter)=elite_fitness; % Para graficar
end

%########################### Chequeos / Ploteos ###########################

for i=1:N % Chequeo que las poblaciones elite sean buenas
    [Fit(i,1),Error(i,1),SalidaReal(:,i)]=fitness(W(:,i),X,Nneu,size_inputs, hidden_layer_amount, size_outputs, V_d);
end

% Es destructivo el crossover? (multiple - runs)
figure(1)
plot(crosses,iter,'o')
title('¿Es destructivo el crossover?')
xlabel('Cantidad de Crossovers')
ylabel('Iteraciones hasta mejor poblacion')
hold on
grid on

figure(2)
plot(fitness_evol)
title('Evolucion del fitness elite de la poblacion')
xlabel('Iteracion')
ylabel('Fitness del individuo elite')
hold on
grid on

% Guardo algunos individuos para probar los .mat en otra red y comprobar si
% realmente funcionan bien 
w=W(:,101); % cambiar n°
ind = 1;
W_ks_test = reshape(w(ind:Nneu*size_inputs),[Nneu,size_inputs]);
ind = ind + Nneu*size_inputs;
b_k_test = reshape(w(ind:(ind+Nneu)-1),[Nneu,1]);
ind = ind + Nneu;

W_jk_test = reshape(w(ind:(ind+Nneu*Nneu)-1),[Nneu,Nneu]);
ind = ind + Nneu*Nneu;
b_j_test = reshape(w(ind:(ind+Nneu*1)-1),[Nneu,1]);
ind = ind + Nneu*1;

W_oj_test = reshape(w(ind:(ind+size_outputs*Nneu)-1),[size_outputs,Nneu]);
ind = ind + Nneu*size_outputs;
b_o_test = reshape(w(ind:(ind+1)-1),[size_outputs,1]);
ind = ind + size_outputs;
%==========================================================================

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