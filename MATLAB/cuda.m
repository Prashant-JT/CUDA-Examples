clear; clc;
% uno;
cuatro;
% seis;
% ocho;

function uno
    a = (0:99);
    b = 2*a;
    c = dot(a, b);

    fprintf('Resultado: %f\n', c);
end

function cuatro
    x = 2.^(0:5);
    y = [0.00000001, 0.00000003, 0.00000017, 0.00000112, 0.00000865, 0.00007434];
    
    plot(x, y);  
    grid();
    xlabel('Tamaño de la matriz cuadrada');
    ylim([-1*10^-5 9*10^-5]);
    ylabel('Tiempo empleado en segundos');
    title('Producto escalar de matrices cuadradas (Secuencial)');    
end

function seis
    x = 2.^(0:5);
    y_cuda = [0.000004, 0.000004, 0.000004, 0.000005, 0.000010, 0.000080];
    
    y_seq = [0.00000001, 0.00000003, 0.00000017, 0.00000112, 0.00000865, 0.00007434];
    
    plot(x, y_cuda, x, y_seq);  
    grid();
    xlabel('Tamaño de la matriz cuadrada');
    ylabel('Tiempo empleado en segundos');
    ylim([-1*10^-5 9*10^-5]);
    title('Comparativa del producto escalar de matrices cuadradas (1..32)');    
    legend('CUDA', 'Secuencial', 'Location', 'NorthWest');
end

function ocho
    x = 2.^(5:10);
    y_cuda = [0.000702, 0.001355, 0.006762, 0.048501, 0.376575, 3.033569];
    
    y_seq = [0.0000800, 0.00054000, 0.00471000, 0.05905000, 0.46531000, 6.28000000];
    
    plot(x, y_cuda, x, y_seq);  
    grid();
    xlabel('Tamaño de la matriz cuadrada');
    ylabel('Tiempo empleado en segundos');
    ylim([-1 6.5]);
    title('Comparativa del producto escalar de matrices cuadradas (32..1024)');    
    legend('CUDA', 'Secuencial', 'Location', 'NorthWest');
end


