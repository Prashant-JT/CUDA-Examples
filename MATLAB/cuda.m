clear; clc;
% uno;
% cuatro;

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
    title('Producto escalar de matrices cuadradas');
end


