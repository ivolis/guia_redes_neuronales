clear all;clc;close all;

% Entrenamiento

fileNames=["panda.bmp" "perro.bmp" "torero.bmp" "v.bmp" "quijote.bmp"]; % "paloma.bmp"
Np=length(fileNames);
Nneu=45*50;

    for i=1:Np
    imagen=imread(fileNames(i));
    imagen=imagen(1:45,1:50); % Recorto usando los minimos hardcodeados
    P(:,i)=imagen(:);
    end


P=2*P-1; % "Linealizo" para que quede como +-1

W=P*P'; % P tiene una imagen en c/columna
W=P*P'-Np*eye(Nneu);


% Ejecucion

S=imread("panda_modif.bmp"); % meto la imagen modificada
S=S(1:45,1:50);
S=S(:);
S=2*S-1;
%S=S*-1; % Para chequear la convergencia del estado negado

step=.00001; % es para visualizar el cambio

% Mezcla ##################################################################
S2=imread("torero.bmp");
S2=S2(1:45,1:50);
S2=2*S2(:)-1;

S3=imread("perro.bmp");
S3=S3(1:45,1:50);
S3=2*S3(:)-1;

% S=sign(S+S2+S3); % estado mezcla
% figure;imagesc((reshape(S,45,50)+1)/2)
%##########################################################################

% Actualizo asincronicamente

index = linspace(1,Nneu,Nneu);
% ahora hago un shuffle para recorrerlo de manera estocástica
index = index(randperm(length(index)));

for i=1:Nneu
    S(index(i))=sign( dot( W(index(i),:) , S ) );
    result=(reshape(S,45,50)+1)/2;
    if mod(i,15)==0 % hace 15 pasos antes de imprimir
        pause(step);imagesc(result); % visualización "temporal" de converg
    end
end

