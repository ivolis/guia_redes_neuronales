close all; clear all;clc;
f = waitbar(0, 'cero porciento'); % creo la barra

n=35;
dipolos=zeros(1,n*n); % Arreglo de n*n dipolos (1D)

for i=1:n*n
    dipolos(i)=i;
end

W=zeros(n*n,n*n); % Inicializo matriz de pesos W

for i=1:n*n
    for j=1:n*n
        if abs(dipolos(i)-dipolos(j))==1
            W(i,j)=1;
        end
    end
end


S=randsrc(n*n,1,[-1 1]);
hExt=0;
k=1;
cant_puntos=50;
Tmax=4;
Tmin=1;
T=linspace(Tmax,Tmin,cant_puntos);
mean_S=zeros(1,cant_puntos);

H1=-0.5*S'*W*S;%-hExt*sum(S); se debería poner pero hExt=0 y realentiza

for j=1:cant_puntos
    cambios=65;% Numero random para flipear varias veces
    while(cambios ~= 0) % flipeo varias veces todos los dipolos aleatoriamente
        rnd=randperm(n*n); % "distintas aleatoreidades"
        for i=1:n*n
            S(rnd(i))=S(rnd(i))*(-1);
%             S(rnd(i))
%             pause;
            H2=-0.5*S'*W*S;
            deltaH=H2-H1;
%             deltaH
%             pause;
            if deltaH > 0
                if (binornd(1,exp(-deltaH/(k*T(j))))==0) 
                    S(rnd(i))=S(rnd(i))*(-1); % no acepto cambio, vuelvo al orig
%                     disp("no cambio")
                else
                    H1=H2;
                end
            else 
                H1=H2;
            end
        end
        cambios=cambios-1;
    end
    mean_S(j)=mean(S);
    waitbar(j/cant_puntos, f, j*100/cant_puntos + "%"); % actualizo la barra
end

p=polyfit(T,abs(mean_S),4) % aproximo la curva por un pol de grado n (4)
% Meramente visual para ver la forma de la curva

figure(1);  % Ploteo abs porque no interesa si se magnetizo pa arriba o pabajo, si no si se magnetizó y ya
plot(T,abs(mean_S),'o')
%fplot(@(x) p(5)+p(4)*x+p(3)*(x.^2)+p(2)*(x.^3)+p(1)*(x.^4),[Tmin Tmax])
xlabel('T')
ylabel('<S>')
ylim([-0.2 1])
set(gca,'xdir','reverse')
grid on;
