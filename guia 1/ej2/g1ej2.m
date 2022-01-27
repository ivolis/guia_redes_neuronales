clear all;clc;close all;

%##########################################################################
%  Inicializaciónes
%##########################################################################

Nneu=50; % cantidad de neuronas inicial
step_neu=50;% step para "puntos de medición"
Perror=0.1; % prob de error objetivo
err=0; % Inicializo el error en 0
i=0; % un index cualquiera que uso para crear vectores

%##########################################################################
%  "Main"
%##########################################################################

while(Nneu<=500)
    p=0; % inicializo p en 0
    while(err<Perror)
        p=p+1; % err<Perror => se banca otro patrón
        P(:,p)=randsrc(Nneu,1,[-1 1; (1-1/2) 1/2]); % Meto vector de +-1 con prob=1/2 en la columna p de P
        W=P*P'-p*eye(Nneu); % diagonal nula
        err=mean( ( signo(W*P)-P ) ~= 0,'all'); % Calculo error total (mi "entrada" es uno de los patrones APRENDIDOS)
    end
    i=i+1;
    C(i)=(p-1)/Nneu; % capacidad
    pMax(i)= p; 
    Neuronas(i)=Nneu;
    Nneu=Nneu+step_neu;
    err=0; % Reinicio el error en 0 (para que entre al while en 0)
    clear P % reinicio la matriz
end


%##########################################################################
mean(C); % esperanza ==> es la media de la capacidad para un Perror fijo
std(C); % desvio standar
%##########################################################################



%##########################################################################
%  Ploteo
%##########################################################################

figure;
plot(Neuronas,pMax,'o',Neuronas,mean(C)*Neuronas+std(C))
title(strcat('Perror=',num2str(Perror)))
legend('Puntos experimentales', strcat( 'Regresión lineal: y=',num2str( mean(C) ),'x+',num2str( std(C) ) ) )
xlabel('Cantidad de neuronas')
ylabel('Patrones máximos almacenados')
grid on;