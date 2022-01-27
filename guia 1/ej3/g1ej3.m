%% ########################################################################
%                       Primera parte
%##########################################################################

clear all;clc;

Nneu=100; % cantidad de neuronas
p=20; % patrones a aprender

for i=1:p
P(:,i)=randsrc(Nneu,1,[-1 1]);
end

% Aprendizaje (creo la matriz de pesos sinápticos)
W=P*P'-p*eye(Nneu);

% Elimino de forma aleatoria (no-simétrica)
cant_puntos=30;
percent_sinap_elim=linspace(0,1,cant_puntos);
cant_sinap_elim=cast(percent_sinap_elim*Nneu*Nneu,'uint32');
% elim=Cant/w_elementos ; w_elementos son neneuXnneu 
% Ya tengo la cantidad de sinapsis que necesito eliminar para ciertos
% porcentajes distintos.


pasadas=10; % hago varias pasadas en cada punto para calcular la media de cada
            % punto que quiero interpolar en el gráfico final (más preciso)

    for i=1:cant_puntos
       for k=1:pasadas
           W_aux=W; % si no elimina en porcentaje acumulativo, laburo sobre la copia
           rnd=randperm(Nneu*Nneu);
           for j=1:cant_sinap_elim(i)
              W_aux(rnd(j))=0; % Elimino un peso random, tratando matriz como fila larga!     
           end
           err(k)=mean( ( signo(W_aux*P)-P ) ~= 0,'all');
       end
       mean_error(i)=mean(err);
       std_error(i)=std(err);
    end



%##########################################################################
%  Ploteo
%##########################################################################

figure(1);
errorbar(percent_sinap_elim,mean_error,std_error,'o')
hold on;
plot(percent_sinap_elim,mean_error)
xlabel('Fracción de sinapsis/interconexiones eliminadas')
ylabel('Error')
grid on;


%% ########################################################################
%                       Segunda parte
%##########################################################################

clear all;clc;

Nneu=100;
Perror=0.01;
p=0; % inicializo p en 0

cant_puntos=20;
percent_sinap_elim=linspace(0,1,cant_puntos);
cant_sinap_elim=cast(percent_sinap_elim*Nneu*Nneu,'uint32');
% Idem punto anterior

pasadas=10; % hago varias pasadas en una para calcular la media de cada
            % punto que quiero interpolar en el gráfico final

for i=1:cant_puntos
    for k=1:pasadas
       rnd=randperm(Nneu*Nneu);
       err=0; % Inicializo el error en 0
       p=0; % Inicializo en 0 loas p_aprendidos
       while(err<Perror) % con x% borrados, cuanto vale p_max?
            p=p+1; 
            P(:,p)=randsrc(Nneu,1,[-1 1; (1-1/2) 1/2]);
            W=P*P'-p*eye(Nneu);
            W_aux=W;
            for j=1:cant_sinap_elim(i)
                W_aux(rnd(j))=0;
            end
            err=mean( ( signo(W_aux*P)-P ) ~= 0,'all');
       end
       C(k)=(p-1)/Nneu; % capacidad
       clear P;
    end
    mean_capacidad(i)=mean(C);
    std_capacidad(i)=std(C);
end

%##########################################################################
%  Ploteo
%##########################################################################

figure(1);
errorbar(percent_sinap_elim,mean_capacidad,std_capacidad,'o')
hold on;
plot(percent_sinap_elim,mean_capacidad)
xlabel('Fracción de interconexiones eliminadas')
ylabel('Capacidad')
grid on;