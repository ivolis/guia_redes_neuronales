%% 
clc;clear all;close all

N=1200; % Entrenamiento
M=400; % Testeo


% Creo los vectores de: 

        % Entrenamiento
x_train = unifrnd(0,2*pi,N,1);
y_train = unifrnd(0,2*pi,N,1);
z_train = unifrnd(-1,1,N,1);

X_train = [x_train,y_train,z_train];
V_d_train=sin(x_train)+cos(y_train)+z_train;

        % Testeo
x_test = unifrnd(0,2*pi,M,1);
y_test = unifrnd(0,2*pi,M,1);
z_test = unifrnd(-1,1,M,1);

X_test = [x_test,y_test,z_test];
V_d_test=sin(x_test)+cos(y_test)+z_test;

% =========================================================================

% DEFINO PARÁMETROS =======================================================
eta=0.003%0.016%0.001;
X=X_train;
V_d=V_d_train;
Nneu=16;
% =========================================================================

% Defino los pesos sinapticos iniciales
media=0;
varianza=0.2;

%-> normrnd(media,varianza,fila,column)
W_ks = normrnd(media,varianza,Nneu,length(X(1,:)));
W_ok = normrnd(media,varianza,1,Nneu);

% Defino los bias
b_k = normrnd(media,varianza,Nneu,1);
b_o = normrnd(media,varianza,1,1);

% Inicializo los deltas y los bias en 0
[dW_ok , dW_ks] = deal(0);
[db_o , db_k] = deal(0);


% Aprendizaje
epoca=0; % Si no encuentra una cvg, lo paro yo.
iteracion=0; % para graficar E_train
max_epoca=550;
E_test=1; % Para que entre al while
V_o_train=zeros(length(V_d),1); % Inicializo las salidas reales en 0.
while(E_test(end)>0.002)
    %Quiero evitar un loop infinito 
    if (epoca==max_epoca)
        break
    end
    epoca=epoca+1;
    
    % Forwarding y Back
    for mu=randperm(length(X)) % recorro aleatoriamente todos los patrones
        iteracion=iteracion+1;
        % Entrada a capa k
        h_k = (W_ks*X(mu,:)') + b_k;
        V_k = tanh(h_k);
        
        % Capa k a capa o
        h_o = (W_ok*V_k) + b_o;
        V_o_train(mu) = tanh(h_o); % Guardo salidas reales para c/patron
        
        E_train(iteracion)=0.5*( (V_d_train(mu)/3) - V_o_train(mu) )^2;
        
        % > Backpropagation
        
        % Desplazamiento pesos oi y bias o
        delta_o = (1-tanh(h_o).^2)*((V_d(mu)/3)-V_o_train(mu));
        dW_ok = eta*delta_o*V_k';
        db_o = eta*delta_o;
        
        % Desplazamiento de pesos ks y bias k
        delta_k = (1-tanh(h_k).^2).*(W_ok'*delta_o);
        dW_ks = eta*delta_k*X(mu,:);
        db_k = eta*delta_k;
        
        % Actualizo para c/patron
        W_ks = W_ks + dW_ks;
        W_ok = W_ok + dW_ok;
        
        b_o = b_o + db_o;
        b_k = b_k + db_k;
    end
   
    % Testeo
    V_o_test=zeros(length(V_d_test),1);
    for mu=randperm(length(X_test))
        % Entrada a capa k
        h_k = (W_ks*X_test(mu,:)') + b_k;
        V_k = tanh(h_k);
        
        % Capa k a capa o
        h_o = (W_ok*V_k) + b_o;
        V_o_test(mu) = tanh(h_o); % salida real de testeo
    end
        
    E_test(epoca)=0.5*sum( ( (V_d_test/3) - V_o_test ).^2 )/M;
    % /M porque uso la MEDIA del error   
    
    E_test(end) % para ver
end

figure('Name', 'error test vs epoca') % E_test vs Epoca
plot(E_test)
title('E_{target}=0.002')
xlabel('Epoca')
ylabel('E_{test}')
grid on

figure('Name', 'error train vs iter') % E_train vs iteracion
plot(E_train)
title(['E_{train} final = ' num2str(E_train(end))])
xlabel('Iteracion')
ylabel('E_{train}')
grid on

figure('Name', 'Salida real vs deseada') % Salida real vs Salida deseada
plot(V_d_test,3*V_o_test,'o')
title(['Error_{test} medio = ' num2str(E_test(end))])
hold on
createfit_2data(V_d_test,3*V_o_test)
xlabel('Salida deseada')
ylabel('Salida real')
grid on