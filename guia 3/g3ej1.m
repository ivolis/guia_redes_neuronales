clc
clear all
close all
% Estimulos (patrones de entrada)

%% Circulo
nombre = ('CIRCULO')
n = 2000; % cantidad de patrones/puntos
R = 1; % radio del circulo (r_max sería)
theta = 2*pi*rand(n,1);
radio = R*sqrt(rand(n,1));
x_1 = radio.*cos(theta);
x_2 = radio.*sin(theta);
% x_1 y x_2 simplemente mapean el punto que quiero (patron 2D)

n_patrones = length(x_1);
media_pesos = 0;
varianza_pesos = 0.15;

%% Clusters
nombre = ('CLUSTER')
n = 2001; % cantidad de patrones/puntos
R = 0.2; % radio del circulo (r_max sería)

rx1 = -0.5;
ry1 = -0.5;
theta = 2*pi*rand(n/3,1);
radio = R*sqrt(rand(n/3,1));
x_1 = rx1 + radio.*cos(theta);
x_2 = ry1 + radio.*sin(theta);

rx2 = 0.5;
ry2 = 0.5;
theta = 2*pi*rand(n/3,1);
radio = R*sqrt(rand(n/3,1));
x_1 = [x_1 ; rx2 + radio.*cos(theta)];
x_2 = [x_2 ; ry2 + radio.*sin(theta)];

rx3 = -0.5;
ry3 = 0.5;
theta = 2*pi*rand(n/3,1);
radio = R*sqrt(rand(n/3,1));
x_1 = [x_1 ; rx3 + radio.*cos(theta)];
x_2 = [x_2 ; ry3 + radio.*sin(theta)];

n_patrones = length(x_1);
media_pesos = 0;
varianza_pesos = 0.24;


%% Cuadrado
nombre = ('CUADRADO')
n = 2000;
x_1 = unifrnd(-1,1,n,1);
x_2 = unifrnd(-1,1,n,1);

n_patrones = length(x_1);
media_pesos = 0;
varianza_pesos = 0.2;

%% Letra L
nombre = ('L')
n = 2000; % Voy a tener menos de n patrones
x_1 = unifrnd(-1,1,2*n,1);
x_2 = unifrnd(-1,1,2*n,1);
iter = length(x_1);
while(iter>0)
    if x_1(iter) > -0.5 && x_2(iter) > -0.5
        x_1(iter)=[];
        x_2(iter)=[];
    end 
    iter = iter - 1;
end

n_patrones = length(x_1);
media_pesos = -0.5;
varianza_pesos = 0.15;

%%

% Espacio Neuronas
neu_y = 21;
neu_x = 21;
Nneu = neu_x*neu_y;

% Espacio de estimulos
entradas=2; % x_1 y x_2

% Aprendizaje
eta = 0.45;
anchos_vecindad=3:-0.1:0.1;

% Pesos sinapticos
W1=normrnd(media_pesos,varianza_pesos,neu_x,neu_y);
W2=normrnd(media_pesos,varianza_pesos,neu_x,neu_y);

% Arranco el proceso de visualización de topologia
figure(1)
scatter(x_1,x_2)
title(['Cantidad de patrones = ' num2str(n_patrones)])
pbaspect([1 1 1]);
hold on
h1 = plot(W1,W2,'k', 'linewidth',2);
h2 = plot(W1',W2','k', 'linewidth',2);
if (nombre == 'CIRCULO')
    viscircles([0,0],[1],'edgecolor','k');
end
grid on
saveas(gcf, ['INICIAL_' nombre '.jpg'])
%pause; % Para guardar la inicial


patron_elegido = zeros(1,2); % inicializo
costo = zeros(1,length(anchos_vecindad));

for iter=1:length(anchos_vecindad)
    ancho_vecindad=anchos_vecindad(iter);
    for j=randperm(n_patrones) % elijo un patron al azar (balanceado)
        patron_elegido(1)=x_1(j);
        patron_elegido(2)=x_2(j);
        distancia_min=1; % numero random "grande"

        % Distancia euclidea entre los pesos y el patron aleatorio
        for i=1:Nneu
            aux=patron_elegido-[W1(i) W2(i)]; % matriz como vector *
            distancia=sqrt(aux*aux');
            if distancia < distancia_min
                distancia_min = distancia;
                neurona_ganadora=i;
            end
        end
        [row,col] = ind2sub([neu_x neu_y],neurona_ganadora);
        r_win = [row,col]; % Neurona ganadora en el espacio de neuronas

        % Vecindad
        for i=1:Nneu
            [row,col] = ind2sub([neu_x neu_y],i);
            r_neu = [row,col];
            dist_neu = (r_neu-r_win)*(r_neu-r_win)';
            vecindad = exp(-dist_neu/(2*ancho_vecindad));

            w = [W1(row,col) , W2(row,col)];
            delta_wij=eta*vecindad*(patron_elegido-w);
            W1(row,col)=W1(row,col) + delta_wij(1);
            W2(row,col)=W2(row,col) + delta_wij(2);
        end
    end
    delete(h1)
    delete(h2)
    h1 = plot(W1,W2,'k', 'linewidth',2);
    h2 = plot(W1',W2','k', 'linewidth',2);
    pause(0.00001)
    if iter == length(anchos_vecindad)/2
        saveas(gcf, ['INTERMEDIO_' nombre '.jpg'])
        %pause;
    end % Para guardar la intermedia
    
    % Calculo de la funcion costo C
    C=0; % Inicializo
    for j=randperm(n_patrones)
        patron_elegido(1)=x_1(j);
        patron_elegido(2)=x_2(j);
        distancia_min=1;

        
        for i=1:Nneu
            aux=patron_elegido-[W1(i) W2(i)];
            distancia=sqrt(aux*aux');
            if distancia < distancia_min
                distancia_min = distancia;
                neurona_ganadora=i;
            end
        end
        [row,col] = ind2sub([neu_x neu_y],neurona_ganadora);
        r_win = [row,col];

        
        for i=1:Nneu
            [row,col] = ind2sub([neu_x neu_y],i);
            r_neu = [row,col];
            dist_neu = (r_neu-r_win)*(r_neu-r_win)';
            vecindad = exp(-dist_neu/(2*ancho_vecindad));

            w = [W1(row,col) , W2(row,col)];
            C = C + vecindad*norm(patron_elegido-w)^2;
        end
    end
    costo(iter) = 1/2*C;
end

saveas(gcf, ['FINAL_' nombre '.jpg'])

figure(2)
plot(costo)
xlabel('Iteracion')
ylabel('Costo')
grid on
saveas(gcf, ['funcion-costo_' nombre '.jpg'])