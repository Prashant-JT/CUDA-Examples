clear; clc;
% uno;
% cuatro;
seis;

function uno
    a = (0:99);
    b = 2*a;
    c = dot(a, b);

    fprintf('Resultado: %f\n', c);
end

function cuatro
    x = 2.^(0:5);
    y = [0.0, 0.0, 0.00000002, 0.00000012, 0.00000114, 0.00001285];
    
    plot(x, y);  
    grid();
    xlabel('Tamaño de la matriz cuadrada');
    ylabel('Tiempo empleado en segundos');
    title('Producto escalar de matrices cuadradas (Secuencial)');    
end

function seis
    x = 2.^(0:5);
    y_cuda = [0.000004, 0.000004, 0.000004, 0.000005, 0.000010, 0.000080];
    
    y_seq = [0.0, 0.0, 0.00000002, 0.00000012, 0.00000114, 0.00001285];
    
    plot(x, y_cuda, x, y_seq);  
    grid();
    xlabel('Tamaño de la matriz cuadrada');
    ylabel('Tiempo empleado en segundos');
    title('Comparativa del producto escalar de matrices cuadradas');    
    legend('CUDA', 'Secuencial', 'Location', 'NorthWest');
end


