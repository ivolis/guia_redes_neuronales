close all;clear all;clc;

% Patrones con una columna de 1s para el bias (w0)
%==========================================================================
%               2 entradas
%==========================================================================
x_2 = [1 -1 -1;1 1 -1;1 -1 1;1 1 1]';                           

y_d_AND_2 = [-1;-1;-1;1];
y_d_OR_2 = [-1;1;1;1];

%==========================================================================
%               4 entradas
%==========================================================================
x_4 = 2*[1,0,0,0,0;
     1,0,0,0,1;
     1,0,0,1,0;
     1,0,0,1,1;
     1,0,1,0,0;
     1,0,1,0,1
     1,0,1,1,0
     1,0,1,1,1
     1,1,0,0,0
     1,1,0,0,1
     1,1,0,1,0
     1,1,0,1,1
     1,1,1,0,0
     1,1,1,0,1
     1,1,1,1,0
     1,1,1,1,1]'-1;
 
y_d_AND_4 = [-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;-1;1];
y_d_OR_4 = [-1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];

%--------------------------------------------------------------------------

%============== Elijo que compuerta voy a usar ============================
n_inputs=4+1; % inputs + w_0 (bias)
y_d=y_d_OR_4; % salida deseada para c/compuerta
x=x_4; % patrones a aprender
%==========================================================================

W=rand(n_inputs,1); % inicializo W con números aleatorios (uniforme) entre
                       % 0 y 1
p=length(x(1,:));
y=zeros(p,1); % salidas posibles
eta=0.8;


E=1; % inicializo con un valor para entrar al while
i=0; % variable que cuenta iteraciones 
while(E>0)
    i=i+1;
    E=0;
    rnd=randperm(p); % así recorro aleatoriamente que x^i agarrar
    
    for mu=1:p
        mu_actual=rnd(mu); % eligo un x aleatorio
        y(mu_actual)=signo( W' * x(:,mu_actual) ); % y
        delta_W=eta*x(:,mu_actual)*( y_d(mu_actual) - y(mu_actual) );
        W=W+delta_W;
    end
    % Calculo el error una vez que termine de actualizar W
    % Es decir, aprende => error => aprende => error. Si meto el error
    % adentro como estaba haciendo antes estoy como calculando el error
    % mientras aprende!
    for mu=1:p
        mu_actual=rnd(mu); % eligo un x aleatorio
        y(mu_actual)=signo( W' * x(:,mu_actual) ); % y^mu 
        E=E+0.5*(y_d(mu_actual)-y(mu_actual))^2;
    end
    errores(i)=E;
end


% ¿Rapidismo?
figure(1)
plot(errores)
title('Evolución del error para de entradas')
xlabel('Iteración')
ylabel('Error')
hold on
grid on

if n_inputs == 3
    possible_inputs=linspace(-2,2); % de -2 a 2 solo para visual de grafico
    figure(2)
    plot(possible_inputs,(-W(1)/W(3))-(W(2)/W(3))*possible_inputs);
    hold on
    plot(1,1,'bO',-1,1,'rO',1,-1,'rO',-1,-1,'rO')
    ylim([-2 2])
    xlim([-2 2])
    grid on
end

%% Estimación de capacidad del perceptron simple
clear all;clc;close all;

f = waitbar(0, 'cero porciento'); % creo la barra

N=20;
p_max=4*N;
N_ap=zeros(1,p_max);
eta=6;
max_iterac_error=500; % para no caer en loops inf, pongo cota de intentos
                                % para aprender los patrones

for p=1:p_max
    Nap=0;
    N_rep=500;
    waitbar(p/p_max, f, p*100/p_max + "%"); % actualizo la barra
    while(N_rep>0)
        % Genero p patrones
        for i=1:p
            for j=1:N
                x(i,j)=unifrnd(-1,1); % [-1,1]
            end
            y_d(i,1)=randsrc(1,1,[1 -1]); % {-1,1}
        end
        
        y=zeros(p,1); % salidas posibles
        W=rand(N,1)'; % Arranco con una W aleatoria
        
        for i=1:max_iterac_error
            E=0; % Para c/rep vuelve a 0 la energía
            rnd=randperm(p); % así recorro aleatoriamente que x^i agarrar
            for mu=1:p
                mu_actual=rnd(mu); % eligo un x^mu aleatorio
                y(mu_actual)=signo( W * x(mu_actual,:)' ); % y
                delta_W=eta*x(mu_actual,:)'*( y_d(mu_actual) - y(mu_actual) );
                W=W+delta_W';
            end
            
            for mu=1:p
                mu_actual=rnd(mu); % eligo un x aleatorio
                y(mu_actual)=signo( W * x(mu_actual,:)' ); % y^mu 
                E=E+0.5*(y_d(mu_actual)-y(mu_actual))^2;
            end
        end
        
        if(E<=0) % Pudo aprender!
        Nap=Nap+1;
        end
        % Si no entra al if no pudo aprender los p patrones
        % (al menos en max_iterac_error)
        N_rep=N_rep-1;
    end
    N_ap(p)=Nap;
end

Nrep=500; % para plotear
p_vector=linspace(1,4*N,4*N); % para plotear

figure(1)
plot(p_vector,N_ap/Nrep)
grid on
