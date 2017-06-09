function Evaluate(deepnet_f)
clearvars('-global');
hF=figure('MenuBar','none','NumberTitle','off',...
    'Name','Paint','Resize','off',...
    'Position',[200 200 560 560]);
centerfig();

axes('Parent',hF,'Units','Normalized',...
    'Position',[-0.01 -0.02 1.02 1.04]);
dim=get(hF,'Position');
xL=[0 dim(3)];
yL=[0 dim(4)];

% Menú Archivo
hmArch=uimenu(hF,'Label','Arquivo');
uimenu(hmArch,'Label','Salvar','Callback',@guardarImg);
uimenu(hmArch,'Label','Testar','Callback',@testImg);
% uimenu(hmArch,'Label','Abrir imagen','Callback',@abrirImg);
uimenu(hmArch,'Label','Sair','Callback','close(gcf)');

% Menú Seleccionar herramienta
hSH=uimenu(hF,'Label','Escolher Ferramenta');
uimenu(hSH,'Label','Lápiz','Callback',@dmodo,'Checked','on');
uimenu(hSH,'Label','Rectángulo','Callback',@dmodo);
uimenu(hSH,'Label','Círculo','Callback',@dmodo);
uimenu(hSH,'Label','Polilinha','Callback',@dmodo);
uimenu(hSH,'Label','Aerógrafo','Callback',@dmodo);
uimenu(hSH,'Label','Texto','Callback',@dmodo);
uimenu(hSH,'Label','Apagar linea','Callback',@borraLinea,'Separator','on');
uimenu(hSH,'Label','Apagar lienzo',...
    'Callback','cla(gca,''reset'');delete(findobj(''type'',''hggroup''));');

% Menú Color
hSC=uimenu(hF,'Label','Escolher Cor');
uimenu(hSC,'Label','Selecionar Cor','Callback',@scolor);
uimenu(hSC,'Label','Cor de fundo','Callback',@lienzoColor);

% Menú Grosor
hSG=uimenu(hF,'Label','Espessura');
uimenu(hSG,'Label','40','checked','on');
uimenu(hSG,'Label','50');
uimenu(hSG,'Label','60');
uimenu(hSG,'Label','80');
set(findobj('Parent',hSG),'Callback',@sgrosor);

% Menú Ayuda
% hMA=uimenu(hF,'Label','Ajuda');
% uimenu(hMA,'Label','Ajuda','Callback',@ayuda);
% uimenu(hMA,'Label','Acerca de...','Callback',@acerca);

% Menú Contextual -> Axes & Surface 
hMCTX=uicontextmenu();
uimenu(hMCTX,'Label','Limpar lienzo','Callback','cla(gca,''reset'')');
%uimenu(hMCTX,'Label','Color del lienzo','Callback',@lienzoColor);
uimenu(hMCTX,'Label','Testar','Callback',@testImg);
%uimenu(hMCTX,'Label','Lienzo mágico','Callback',@lienzomagFcn);
set(gca,'uicontextmenu',hMCTX);


% ========================== FUNCIONES ==================================

% Llama al modo dibujo 
    function dmodo(src,~)
        hb=findobj('callback',@dmodo);
        set(hb,'checked','off');
        set(src,'checked','on');
        tipo=get(src,'Label');
        dibujar(tipo); % Llamada a la función dibujar, pasando como
		               % argumento el tipo de dibujo.
    end

% Selecciona el tipo de dibujo
    function dibujar(tipo)
        global clr % Color del trazo
        if isempty(clr) 
            clr='k'; % Color negro por defecto
        end
        grs=eval(get(findobj('parent',hSG,'-and','checked','on'),'Label')); % Grosor
        set(hF,'WindowButtonDownFcn',''); % 'Desactivar' eventos para evitar conflictos.
        set(hF,'WindowButtonMotionFcn','');
        set(hF,'WindowButtonUpFcn','');
        if strcmpi(tipo,'Lápiz')
            set(hF,'WindowButtonDownFcn',{@lapizFcn,clr,grs});
        elseif strcmpi(tipo,'Rectángulo')
            set(hF,'WindowButtonDownFcn',{@rectanguloFcn,clr,grs});
        elseif strcmpi(tipo,'Círculo')
            set(hF,'WindowButtonDownFcn',{@circuloFcn,clr,grs});
        elseif strcmpi(tipo,'Polilínea')
            set(hF,'WindowButtonDownFcn',{@polilineaFcn,clr,grs});
        elseif strcmpi(tipo,'Aerógrafo')
            set(hF,'WindowButtonDownFcn',{@aerografoFcn,clr,grs});
        elseif strcmpi(tipo,'Texto')
            set(hF,'WindowButtonDownFcn',{@textoFcn,clr});
        else
            % Pass: Por definir 
        end
    end

% Herramienta Lápiz
    function lapizFcn(src,~,clr,grs)
        if strcmp(get(src,'SelectionType'),'normal')
            set(src,'WindowButtonMotionFcn',@mov,...
                'WindowButtonUpFcn',@bprss);
        end
        function mov(src,~)
            global P
            P=[P;get(src,'CurrentPoint')];
            hold on
            plot(P(:,1),P(:,2),'color',clr,'linewidth',grs);
            axis([xL(1) xL(2) yL(1) yL(2)]);
        end
        function bprss(src,~)
            set(src,'WindowButtonMotionFcn','');
            clearvars('-global','P');
        end
    end

% Herramienta Rectángulo
    function rectanguloFcn(src,~,clr,grs)
        if strcmp(get(src,'SelectionType'),'normal')
            po=get(src,'CurrentPoint');
            set(src,'WindowButtonMotionFcn',{@mov,po},...
                'WindowButtonUpFcn',@bprss);
        end
        function mov(src,~,po)
            delete(findobj('tag','temp'));
            pa=get(src,'CurrentPoint');
            XX=[po(1) po(1) pa(1) pa(1) po(1)];
            YY=[po(2) pa(2) pa(2) po(2) po(2)];
            line(XX,YY,'tag','temp','color',clr,'linewidth',grs);
            axis([xL(1) xL(2) yL(1) yL(2)]);
        end
        function bprss(src,~)
            set(findobj('tag','temp'),'tag','dib');
            set(src,'WindowButtonMotionFcn','');
        end
    end

% Herramienta círculo
    function circuloFcn(src,~,clr,grs)
        set(gca,'NextPlot','add');
        if strcmp(get(src,'SelectionType'),'normal')
            po=get(src,'CurrentPoint');
            set(src,'WindowButtonMotionFcn',{@mov,po},...
                'WindowButtonUpFcn',@bprss);
        end
        function mov(src,~,po)
            delete(findobj('tag','temp'));
            pa=get(src,'CurrentPoint');
            r=sqrt((pa(1)-po(1))^2+(pa(2)-po(2))^2);
            t=0:pi/180:2*pi;
            x=r*cos(t)+po(1);
            y=r*sin(t)+po(2);
            plot(x,y,'tag','temp','color',clr,'linewidth',grs);
            axis([xL(1) xL(2) yL(1) yL(2)]);
        end
        function bprss(src,~)
            % Asignar el tag 'dib' (en lugar de 'temp')al último trazo
            % para evitar que sea borrado en procedimientos posteriores.
            set(findobj('tag','temp'),'tag','dib');
            set(src,'WindowButtonMotionFcn',''); % 'Desconectar' evento de 
                                                 % movimiento del mouse.
        end
    end

% Herramienta Polilínea
    function polilineaFcn(src,~,clr,grs)
        global XPL YPL
        set(gca,'NextPlot','add');
        axis([xL(1) xL(2) yL(1) yL(2)]);
        P=get(src,'CurrentPoint');
        XPL=[XPL P(1)];
        YPL=[YPL P(2)];
        if strcmp(get(src,'SelectionType'),'normal')
            plot(XPL,YPL,'color',clr,'linewidth',grs);
            set(src,'WindowButtonMotionFcn',@movpoli);
        elseif strcmp(get(src,'SelectionType'),'alt')
            plot(XPL,YPL,'color',clr,'linewidth',grs);
            clearvars('-global','XPL','YPL');
            set(src,'WindowButtonMotionFcn','');
        else
            % Pass: Por definir
        end
        
        function movpoli(src,~)
            delete(findobj('tag','temp'));
            paux=get(src,'CurrentPoint');
            plot([XPL(end) paux(1)],[YPL(end) paux(2)],...
                'color',clr,'linewidth',grs,'tag','temp');
        end
    end

% Herramienta Aerógrafo
    function aerografoFcn(src,~,clr,grs)
        set(gca,'NextPlot','add');
        if strcmp(get(src,'SelectionType'),'normal')
            set(src,'WindowButtonMotionFcn',@mov,...
                'WindowButtonUpFcn',@bprss);
        else
            % Pass: A definir en versiones posteriores
        end
        function mov(src,~)
            P=get(src,'CurrentPoint');
            kg=2*grs;
            XX=randi([P(1)-kg P(1)+kg],1,10);
            YY=randi([P(2)-kg P(2)+kg],1,10);
            plot(XX,YY,'.','MarkerSize',2,...
                'MarkerEdgeColor',clr);
            axis([xL(1) xL(2) yL(1) yL(2)]);
        end
        function bprss(src,~)
            set(src,'WindowButtonMotionFcn','');
        end
    end

% Inserta un texto
    function textoFcn(src,~,clr)
        set(gca,'NextPlot','add');
        if strcmp(get(src,'SelectionType'),'normal')
            p=get(src,'CurrentPoint');
            txt=cell2mat(inputdlg('Texto a insertar'));
            annotation('textbox','units','pixels',...
                'Position',[p(1) p(2) 10 10],'String',txt,...
                'LineStyle','none','FitBoxToText','on',...
                'Color',clr);
        else
            % Pass
        end
    end

% Lienzo mágico: Color de fondo random (Múltiples Colores)
    function lienzomagFcn(~,~)
        set(gca,'NextPlot','add');
        delete(findobj('type','surface')); % Borra objetos de tipo 'Surface' para
		                                   % evitar una posible ralentización.
        axis([xL(1) xL(2) yL(1) yL(2)]); % Límites pre-establecidos
		
        % Definición de mapas de colores personalizados ================================
        map1=rand(64,3); % Mapa de color aleatorio
        map2=repmat(gray(3)*0.8,8,1); % Escala de grises 'modificada'
        map3=repmat([1 1 0;0 1 1;1 0 1;1 0 0;0 1 0;0 0 1],8,1); % Franjas de colores
        %  =============================================================================
        [xx,yy]=meshgrid(linspace(0,10,xL(2)),linspace(0,10,yL(2)));
        clrmaps={'hsv','hot','jet','cool','winter',...
            'summer','autumn',map1,map2,map3};
        funs={'xx+yy','sin(xx)+cos(yy)','xx.^2+yy.^2','xx./(yy+10)',...
            'sin(xx).*yy/2','(xx+yy).^(cos(yy)+sin(xx))','xx','yy'};
        zz=eval(funs{randi([1 8])});
        hpc=pcolor(zz);
        colormap(clrmaps{randi([1 10])});
        shading('interp');
        set(hpc,'uicontextmenu',hMCTX);
    end

% Borra linea seleccionada
    function borraLinea(~,~)
        set(hF,'WindowButtonDownFcn',@selectLinea);
        function selectLinea(~,~)
            if strcmp(get(gco,'type'),'line')
                delete(gco);
            end
        end
    end

% Color de lienzo
    function lienzoColor(~,~)
        delete(findobj('type','surface'));
        lcol=uisetcolor();
        set(gca,'color',lcol);
    end

% Selector de color
    function scolor(~,~)
        global clr
        clr=uisetcolor();
        dmodo(findobj('parent',hSH,'-and','checked','on'));
    end
% Selector de grosor
    function sgrosor(src,~)
        hb=findobj('callback',@sgrosor);
        set(hb,'checked','off');
        set(src,'checked','on');
        dmodo(findobj('parent',hSH,'-and','checked','on'));
    end

% Función guardar
    function guardarImg(~,~)
        [nombre,direct]=uiputfile({'*.png'},'Guardar dibujo');
        if isequal(nombre,0) || isequal(direct,0)
            return;
        else
            IMG=getframe(gca);
            I = rgb2gray(IMG.cdata);
            IMG = imcomplement(I);
            imshow(IMG)
            IMG28 = imresize(IMG, [28, 28]);
            imwrite(IMG28,fullfile(direct,nombre),'png');
            helpdlg('Operación terminada','Paint');
        end
    end
% Función Testar
    function testImg(~,~)         
        IMG=getframe(gca);
        I = rgb2gray(IMG.cdata);
        IMG = imcomplement(I);
        %imshow(IMG)
        IMG28 = imresize(IMG, [28, 28]);
        dato1=double(IMG28)/255;    
        res1=deepnet_f(dato1(:))'; 
        
%         figure(2)
%         imagesc(reshape(dato1(:),28,28)),colormap gray
        [B1,I1] = sort(res1);
        I1(I1==10)=0;
%        I1=I1-1;
        tex1=strcat(mat2str(I1(10)), ' ---> ',mat2str(B1(10)*100,5),'%');
        tex2=strcat(mat2str(I1(9)), ' ---> ',mat2str(B1(9)*100,5),'%');
        tex3=strcat(mat2str(I1(8)), ' ---> ',mat2str(B1(8)*100,5),'%');
        tex4=strcat(mat2str(I1(7)), ' ---> ',mat2str(B1(7)*100,5),'%');
        helpdlg({'Resultados :' tex1 tex2 tex3 tex4},'Teste');
    end
% Ayuda 
    function ayuda(~,~)
        if exist('Ayuda.html','file')
            web('Ayuda.html','-browser');
        else
            errordlg('Archivo de ayuda no encontrado','Paint');
        end
    end

% Acerca de ...
    function acerca(~,~)
        figure('MenuBar','none','NumberTitle','off',...
            'Name','Acerca de...','Resize','off',...
            'Position',[0 0 200 100],'color','w');
        centerfig();
        devel='Por: Jorge De Los Santos';
        e_mail='E-mail: delossantosmfq@gmail.com';
        blog='Blog: http://matlab-typ.blogspot.mx';
        nvrs='Paint 0.1.1';
        uicontrol('style','text','String',devel,...
            'Units','Normalized','Position',[0.1 0.80 0.8 0.15],...
            'FontName','Arial Narrow','FontSize',10,...
            'ForegroundColor',ones(1,3)*0.2);
        uicontrol('style','text','String',{e_mail,blog},...
            'Units','Normalized','Position',[0.1 0.45 0.8 0.3],...
            'FontName','Arial Narrow','FontSize',9,...
            'ForegroundColor',ones(1,3)/2);
        uicontrol('style','text','String',nvrs,...
            'Units','Normalized','Position',[0.1 0.15 0.8 0.1],...
            'FontName','Courier','FontSize',10,'FontWeight','b',...
            'ForegroundColor',[0 0 0.5]);
        set(findobj('style','text'),'BackgroundColor','w');
    end
end