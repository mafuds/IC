% Método Maximizer of Posterior Marginals (MPM) de filtragem

% Este método recebe uma imagem ruidosa, o número de iterações desejado, o número de iterações para se atingir o equilíbrio da Cadeia de Markov,
% o modelo Markoviano a priori (GMRF, GIMLL, Potts) e a função de densidade de probabilidade da verossimilhança, e retorna uma
% estimativa MPM da imagem livre de ruído.

% Autoria de Artur de Freitas Mafud, Universidade Estadual Paulista "Júlio de Mesquita Filho", 2015

function [filtrado] = MPM (imgRuidosa,iter,K,modelo,probVero)

filtrado = zeros(size(imgRuidosa)); % zera a matriz final com o tamanho da imagem ruidosa

% Estimativa da imagem livre de ruído
h = fspecial('average');
I = imfilter(uint8(imgRuidosa), h);
I = double(I) + 1;
%

[m,n] = size(I);

% vê qual o modelo Markoviano escolhido
tipo = strcmp(modelo,'GMRF');
tipo2 = strcmp(modelo,'GIMLL');
tipo3 = strcmp(modelo,'Potts');
%

% Calculo do Beta
if (tipo == 1)
    mi = mean(I(:));
    vBeta = funcaoBeta_GMRF(I,mi);
elseif (tipo2 == 1 || tipo3 == 1)
    global somaEnergia;
    global accum;
    accum = zeros(m,n,256);
    somaEnergia = 0;
    fun = @funcaoBeta;
    x0 = 0.1; % "chute" inicial
    valorBeta = fzero(fun,x0);
end
%

cont = 0;

armazena = zeros(m,n,256);

while (cont < iter)
    if (tipo == 1)
        I = padarray(I,[1 1], 'symmetric'); % fazendo padding para conseguir pegar janela na borda
        priori = zeros(m,n,256);
        
        for i = 2 : m + 1
            for j = 2 : n + 1
                janela = I(i-1:i+1, j-1:j+1);
                mi_s = mean(janela(:));
                sigma_s = var(janela(:));
                fracao = 1 / sqrt(2 * pi * sigma_s);
                for cl = 1 : 256
                    expoente = (- 1 / (2 * sigma_s)) * (cl - mi_s - (vBeta * (sum(janela(:) - mi_s) - (I(i,j) - mi_s)))) ^ 2;
                    priori(i-1,j-1,cl) = fracao * exp(expoente);
                end
            end
        end
        
        MAP = priori * probVero;
        
        I = I(2:m+1, 2:n+1);
        
        z = unidrnd(256,[m,n]);
        
        aleatorio = rand(m,n);
        
        [Y, X] = meshgrid(1:m, 1:n);
        
        idx = sub2ind(size(MAP), X(:), Y(:), z(:));
        num = reshape(MAP(idx), [m n]);
        
        idx = sub2ind(size(MAP), X(:), Y(:), I(:));
        den = reshape(MAP(idx), [m n]);
        
        div = num ./ den;
       
        p = min(1, div);
        
        idx = aleatorio <= p;
        
        I(idx) = z(idx);
        
%         for i = 1 : m
%             for j = 1 : n
%                 num = MAP(i, j, z(i,j));
%                 den = MAP(i, j, I(i,j));
%                 div = num / den;
%                 p = min(1,div);
%                 aleatorio = rand;
%                 if (aleatorio <= p)
%                     I(i,j) = z(i,j);
%                 end
%             end
%         end
        
        if (cont > K)
            [Y, X] = meshgrid(1:m, 1:n);
            armazena = armazena + accumarray([X(:), Y(:), I(:)], 1, [m n 256]);
        end
        
        cont = cont + 1
        
    elseif (tipo2 == 1)
        [Y, X] = meshgrid(1:m, 1:n); % organizou matricialmente
        
        for i = -1 : 1
            for j = -1 : 1
                
                if (i == 0 && j == 0)
                    continue;
                end
                
                Imovida = circshift(I,[i j]);
                expoente = -(I - Imovida) .^ 2;
                somaEnergia = somaEnergia + sum(1 - (2 * exp(expoente(:))));
                for cl = 1 : 256
                    accum = accum + accumarray([X(:), Y(:), repmat(cl, size(X(:)))],(1 - (2 * exp(-(cl - Imovida(:)) .^ 2))),[m n 256]);
                end
            end
        end
        energia = exp(valorBeta * accum);
        priori = energia ./ repmat(sum(energia, 3), [1 1 256]);
        
      	MAP = priori * probVero;
        
        z = unidrnd(256,[m,n]);
        
        aleatorio = rand(m,n);
        
        [Y, X] = meshgrid(1:m, 1:n);
        
        idx = sub2ind(size(MAP), X(:), Y(:), z(:));
        num = reshape(MAP(idx), [m n]);
        
        idx = sub2ind(size(MAP), X(:), Y(:), I(:));
        den = reshape(MAP(idx), [m n]);
        
        div = num ./ den;
       
        p = min(1, div);
        
        idx = aleatorio <= p;
        
        I(idx) = z(idx);

	if (cont > K)
            [Y, X] = meshgrid(1:m, 1:n);
            armazena = armazena + accumarray([X(:), Y(:), I(:)], 1, [m n 256]);
        end
        
	cont = cont + 1        

    elseif (tipo3 == 1)
        [Y, X] = meshgrid(1:m, 1:n); % organizou matricialmente
        
        for i = -1 : 1
            for j = -1 : 1
                
                if (i == 0 && j == 0)
                    continue;
                end
                
                Imovida = circshift(I,[i j]);
                TESTE = I == Imovida;
                somaEnergia = somaEnergia + sum(TESTE(:));
                accum = accum + accumarray([X(:), Y(:), Imovida(:)], 1, [m n 256]);
            end
        end
        energia = exp(valorBeta * accum);
        priori = energia ./ repmat(sum(energia, 3), [1 1 256]);
        
        MAP = priori * probVero;
        
        z = unidrnd(256,[m,n]);
        
        aleatorio = rand(m,n);
        
        [Y, X] = meshgrid(1:m, 1:n);
        
        idx = sub2ind(size(MAP), X(:), Y(:), z(:));
        num = reshape(MAP(idx), [m n]);
        
        idx = sub2ind(size(MAP), X(:), Y(:), I(:));
        den = reshape(MAP(idx), [m n]);
        
        div = num ./ den;
       
        p = min(1, div);
        
        idx = aleatorio <= p;
        
        I(idx) = z(idx);

	if (cont > K)
            [Y, X] = meshgrid(1:m, 1:n);
            armazena = armazena + accumarray([X(:), Y(:), I(:)], 1, [m n 256]);
        end
        
	cont = cont + 1
    end
end
armazena = 1 / (iter - K) * armazena;
[M, I] = max(armazena,[],3);
filtrado = I - 1;
end
