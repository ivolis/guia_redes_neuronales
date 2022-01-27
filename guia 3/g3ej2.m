close all
clear all


Nneu = 300;
entradas=2; % x_1 y x_2
eta = 0.5;

% Estimulos (patrones de entrada) - CIUDADES
ciudades = 200
patrones = unifrnd(0,1,ciudades,entradas);
x_1=patrones(:,1);
x_2=patrones(:,2);

% Pesos sinapticos
media_pesos = 0.5;
varianza_pesos = 0.2;
W1=normrnd(media_pesos,varianza_pesos,Nneu,1);
W2=normrnd(media_pesos,varianza_pesos,Nneu,1);

figure(1)
pbaspect([1 1 1]);
scatter(x_1,x_2)
hold on
hpesos = plot(W1,W2,'or');
h1 = plot(W1,W2,'k', 'linewidth',1);
h2 = plot(W1',W2','k', 'linewidth',1);
grid on
saveas(gcf, 'INICIAL_SALESMAN.jpg')

patron_elegido = zeros(1,2); % inicializo
anchos_vecindad=20:-0.1:0.1;
costo = zeros(1,length(anchos_vecindad));

for iter=1:length(anchos_vecindad)
    ancho_vecindad=anchos_vecindad(iter);
    for j=randperm(ciudades) % elijo un patron al azar (balanceado)
        patron_elegido(1)=x_1(j);
        patron_elegido(2)=x_2(j);
        distancia_min=1; % numero random "grande"

        % Distancia euclidea entre los pesos y el patron aleatorio
        for i=1:Nneu
            aux=patron_elegido-[W1(i) W2(i)];
            distancia=sqrt(aux*aux');
            if distancia < distancia_min
                distancia_min = distancia;
                neurona_ganadora=i;
            end
        end
        r_win = neurona_ganadora;

        % Vecindad
        for i=1:Nneu
            r_neu = i;
            dist_neu = (r_neu-r_win)*(r_neu-r_win)';
            
            % Conecto primera y ultima neurona
            if (r_win==1 && i==Nneu) || (r_win==Nneu && i==1)
                dist_neu = 1;
            end
            
            vecindad = exp(-dist_neu/(2*ancho_vecindad));

            w = [W1(i) , W2(i)];
            delta_wij=eta*vecindad*(patron_elegido-w);
            W1(i)=W1(i) + delta_wij(1);
            W2(i)=W2(i) + delta_wij(2);
        end
    end
    
    
    delete(h1)
    delete(h2)
    delete(hpesos)
    hpesos = plot(W1,W2,'or');
    h1 = plot(W1,W2,'k', 'linewidth',1);
    h2 = plot(W1',W2','k', 'linewidth',1);
    pause(0.00001)
    if iter == length(anchos_vecindad)/2
        saveas(gcf,'INTERMEDIO_SALESMAN.jpg')
        %pause;
    end
    
    % Calculo de la funcion costo C
    C=0; % Inicializo
    for j=randperm(ciudades)
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
        r_win = neurona_ganadora;

        for i=1:Nneu
            r_neu = i;
            dist_neu = (r_neu-r_win)*(r_neu-r_win)';
            
            
            if (r_win==1 && i==Nneu) || (r_win==Nneu && i==1)
                dist_neu = 1;
            end
           
            vecindad = exp(-dist_neu/(2*ancho_vecindad));
            w = [W1(i) , W2(i)];
            C = C + vecindad*norm(patron_elegido-w)^2;
        end
    end
    costo(iter) = 1/2*C;
end

saveas(gcf, 'FINAL_SALESMAN.jpg')

figure(2)
plot(costo)
xlabel('Iteracion')
ylabel('Costo')
grid on
saveas(gcf,'funcion-costo_SALESMAN.jpg')