% Método Iterated Conditional Modes (ICM) para filtragem contextual

% Este algoritmo recebe como parâmetros uma imagem ruidosa (formato double), o número de iterações desejado, o modelo Markoviano escolhido (GMRF, GIMLL, Potts) e a função de densidade de probabilidade de verossimilhança, e retorna uma estimativa para Maximum a Posteriori (MAP) da imagem
% livre de ruído

% Autoria de Artur Mafud, Universidade Estadual Paulista "Júlio de Mesquita Filho", 2015

function [filtrado] = ICM (imgRuidosa,iter,modelo,probVero)

filtrado = zeros(size(imgRuidosa)); % zera a matriz final com o tamanho da imagem ruidosa

h = fspecial('average');
I = imfilter(uint8(imgRuidosa), h);

I = double(I) + 1;

[m,n] = size(I);

cont = 0; % contador para o número de iterações

% vê qual o modelo Markoviano escolhido
tipo = strcmp(modelo,'GMRF');
tipo2 = strcmp(modelo,'GIMLL');
tipo3 = strcmp(modelo,'Potts');

while (cont < iter)
    if (tipo == 1)
	I = padarray(imgRuidosa,[1 1], 'symmetric'); % fazendo padding para conseguir pegar janela na borda
        mi = mean(I(:));
        vBeta = funcaoBeta_GMRF(I,mi);
        priori = zeros(m,n,256);
        for i = 2 : m + 1
            for j = 2 : n + 1
                % GMRF
                janela = I(i-1:i+1, j-1:j+1);
                mi_s = mean(janela(:));
                sigma_s = var(janela(:));
                fracao = 1 / sqrt(2 * pi * sigma_s);
                for cl = 1 : 256
                    expoente = (- 1 / (2 * sigma_s)) * (cl - mi_s - (vBeta * (sum(janela(:) - mi_s) - (I(i,j) - mi_s)))) ^ 2;
                    priori(i,j,cl) = fracao * exp(expoente);
                end
                % FIM
            end
        end
        MAP = probVero * priori; % obtendo a posteriori
        cont = cont + 1;
    elseif (tipo2 == 1)
        % CALCULO DA ENERGIA
        global somaEnergia;
        global accum;
        accum = zeros(m,n,256);
        somaEnergia = 0;
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
        % FIM
        
        % ESTIMACAO BETA
        fun = @funcaoBeta;
        x0 = 0.1; % "chute" inicial
        valorBeta = fzero(fun,x0);
        % FIM
        
        % GIMLL
        energia = exp(valorBeta * accum);
        priori = energia ./ repmat(sum(energia, 3), [1 1 256]);
        % FIM
        MAP = probVero * priori; % obtendo a posteriori
        cont = cont + 1;
    elseif (tipo3 == 1)
        % CALCULO DA ENERGIA
        global accum;
        accum = zeros(m, n, 256); % zerou
        [Y, X] = meshgrid(1:m, 1:n); % organizou matricialmente
        global somaEnergia;
        somaEnergia = 0;
        
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
        % ESTIMACAO BETA
        fun = @funcaoBeta;
        x0 = 0.1; % "chute" inicial
        valorBeta = fzero(fun,x0)
        % FIM
        
        % POTTS
        energia = exp(valorBeta * accum);
        priori = energia ./ repmat(sum(energia, 3), [1 1 256]);
        % FIM
        MAP = probVero * priori; % obtendo a posteriori
        cont = cont + 1;
    end
    [M, I] = max(MAP,[],3); % obtendo o indice com a maior probabilidade
end
filtrado = I - 1;
end
