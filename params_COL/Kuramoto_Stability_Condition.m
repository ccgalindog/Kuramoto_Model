clear all; close all; clc;

global N omega

% %% Parámetros del sistema:
% 
% % Número de osciladores
% N = 200; 
% % Frecuencia natural de oscilación de cada nodo distribuida uniformemente
% omega=sqrt(12)*(0.5 - rand(N,1)); 
% % Fuerza de interacción
% K_strength = 0.7;
% % Número de conexiones promedio
% kmedio = 10;  
% % Probabilidad de conexión
% p_conn = kmedio/N; 
% 
% %% Matriz de Conectividad
% % Inicializo la matriz de conectividad
% A = zeros(N);
% 
% % Creo la matriz de conectividad.        
% for i = 1:N
%     for j = i+1:N
%         r = rand;
%         if(r<p_conn)
%             % La matriz es simétrica en este caso particular.
%             A(i,j)=K_strength;
%             A(j,i)=K_strength;
%         end
%     end
% end
% 
% % Para evitar problemas encuentro los nodos que quedaron desconectados
% notConn = find(sum(A~=0)==0);
% % Y elijo aleatoriamente vecinos para conectarlos al menos a un nodo.
% randConn = randperm(N,length(notConn));
% 
% % Aquí me aseguro que cada nodo desconectado se junte con un vecino
% % aleatorio:
% 
% for i = 1:length(notConn)
%     A(notConn(i),randConn(i))=K_strength;
%     A(randConn(i),notConn(i))=K_strength;   
% end    
% 
% 
% 


A = 1.*dlmread('K_Colombia_pu.txt');
omega = dlmread('P_Colombia_pu.txt');
N = length(omega);




% Defino el Laplaciano, para ello calculo primero el grado de cada nodo
degMat = diag(sum(A,2));
% Y el Laplaciano es
Lap = degMat - A;

%% Condición de Sincronización

% Calculo el punto de equilibrio. Para ello calculo primero la 
% pseudoinversa del laplaciano.
Lcross = pinv(Lap);

% Calculo la matriz de incidencia:
B = adj2inc(A);
B = B';

% Calculo el estado estacionario:
thetaSteady = Lcross*omega;

% Calculo la peor distancia del estado estacionario
% que es la que cuyo seno debe ser menor que 1 en el 
% caso de Kuramoto para ser estable.
theta_diff_worst = norm(B'*thetaSteady,Inf)



%% Dinámica

% Defino las condiciones iniciales y tiempo de simulación
theta0 = 2*pi*rand(N,1);
tmax = 50;

options = odeset('InitialStep',1e-8,'MaxStep',1e-7,'RelTol',1e-6,'AbsTol',1e-6.*ones(1,N));
[t,theta] = ode45(@(t,x)kuramoto_model(t,x,A),[0 tmax],theta0);

% Elijo un subconjunto aleatorio de elementos para dibujar
% unitsPlot = randperm(N,N/10);
unitsPlot = 1:N;
thetaPlot = theta(:,unitsPlot);

% Calculo el parámetro de orden r y la fase promedio phase_0
order_param=1/N*sum(exp(1i*theta),2);
r = abs(order_param);
phase_0 = angle(order_param);

% Dibujo tanto la evolución del parámetro de orden como de las fases
figure
subplot(2,1,1)
plot(t,r)
subplot(2,1,2)
hold all
plot(t,wrapToPi(unwrap(thetaPlot)-phase_0))
ylabel('\Delta \theta_i')
ylim([-pi pi])

% figure
% for i = 1:5:length(t)    
%     polar(theta(i,:),1*ones(1,N),'o')
%     legend(['time = ' num2str(t(i))])
%     pause(0.01)
% end

function y = kuramoto_model(t, x,A)

global N omega

theta = x(1:N);
x_dot = omega+sum(A.*sin(meshgrid(theta)-meshgrid(theta)'),2);
y = x_dot;
end
