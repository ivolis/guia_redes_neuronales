clear all;

patrones_train=load('datosTrain.mat');
patrones_train=patrones_train.data';
patrones_train=patrones_train/255; % normalizo (max o 255??)

% ¿Que significa esta matriz?
% 20 imagenes por numero (de 0 a 9) => 10x20=200 columnas/patrones
% Cantidad de pixeles en cada patron (resolucion)
cantidad_patrones = length(patrones_train(1,:));
resolucion = length(patrones_train(:,1));

Nneu = 10; % neuronas en la capa oculta
epsilon=0.1;
iteraciones = 5000;

% Inicializo los pesos sinápticos
media = 0;
varianza = 1;

 % pixeles de los patrones
W_ij = normrnd(media,varianza,resolucion,Nneu);
%Mat de Nneu X (Cantidad de pixeles de los patrones==neuronas capa visible)

% Defino los bias
b_j = normrnd(media,varianza,Nneu,1); % oculta
b_i = normrnd(media,varianza,resolucion,1); % visible

% Entrenamiento
E = zeros(iteraciones,1);


for k=1:iteraciones
    [vh_data,vh_recon,v_datos,v_recons,h_datos,h_recons] = deal(0); % inic
    E_aux=0;
    for mu=1:length(patrones_train(1,:)) % recorro los patrones
        
        % Data
        v_data=normrnd(patrones_train(:,mu),1)';
        
        x = b_j' + v_data*W_ij; % arg de sigma
        h_data=binornd(1,logistic(x));
        
        % Recon
        m_recon = b_i' + h_data*W_ij';
        v_recon = normrnd(m_recon,1);
        
        x = b_j' + v_recon*W_ij; % arg de sigma
        h_recon=binornd(1,logistic(x));
        
        % Voy acumulando para hacer el promedio
        vh_data = vh_data + v_data'*h_data;
        vh_recon = vh_recon + v_recon'*h_recon;
        v_datos= v_datos + v_data;
        v_recons= v_recons + v_recon;
        h_datos= h_datos + h_data;
        h_recons= h_recons + h_recon;
        E_aux=E_aux+mean(abs(patrones_train(:,mu)-m_recon')); % no es especif el error..
    end
    
    E(k)=E_aux/cantidad_patrones; % este es el error (prom de las medias)
    disp(E(k));
    
    delta_Wij=epsilon*(vh_data-vh_recon)/cantidad_patrones;
    delta_bi=epsilon*(v_datos-v_recons)/cantidad_patrones;
    delta_bj=epsilon*(h_datos-h_recons)/cantidad_patrones;
    
    % Actualizo los pesos
    W_ij = W_ij + delta_Wij;
    b_i=b_i+delta_bi';
    b_j=b_j+delta_bj';
end

figure(1)
plot(E)
title(['Error final ' num2str(E(end))])
xlabel('Iteracion')
ylabel('Error')
grid on

%% test
patrones_test = load('datosTest.mat');%Matriz con patrones en las columnas
patrones_test = patrones_test.data'/255;


for i=1:length(patrones_test(1,:))
        v_data = patrones_test(:,i)';
        Vdata_reshape = reshape(v_data,28,28);
        
        imagesc(Vdata_reshape'); % imagen a testear
        axis off;
        saveas(gcf,[num2str(i) '_patron.jpg']) 
        
        %pause;
        x = b_j' + v_data*W_ij;
        H = 1./(1+exp(-x));
 
        
        h_recon = H;
        M_recon = b_i' + h_recon*W_ij';
        M_recon_reshape = reshape(M_recon,[28,28]);
        
        imagesc(M_recon_reshape') % imagen reconstruida
        axis off;
        saveas(gcf,[num2str(i) '_recon.jpg'])
        
        %pause;
end