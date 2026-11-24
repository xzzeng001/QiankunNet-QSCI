figure

nn=length(y1);
x1=1:nn;
nn2=length(y2);
x2=1:nn2;
nn3=length(y3);
x3=1:nn3;
nn4=length(y4);
x4=1:nn4;


yy1=smoothdata(y1);
yy2=smoothdata(y2);
yy3=smoothdata(y3);
yy4=smoothdata(y4);


y_cisd=-1.070884382279194540e+02;
y_fci=-107.43849085267868;

subplot(1,2,1)

hold on
h1=semilogx([1 1e6],[y_cisd,y_cisd],'--','linewidth',2);
col1=get(h1,'color');

h2=plot(x1(1:1:end),yy1(1:1:end),'-o','linewidth',2,'Markersize',12);
col2=get(h2,'color');
%h4.Color(4)=0.8;
set(h2,'MarkerFaceColor',col2);

h3=semilogx(x2(1:1:end),yy2(1:1:end),'-o','linewidth',2,'Markersize',12);
col3=get(h3,'color');
set(h3,'MarkerFaceColor',col3);

h4=semilogx(x3(1:1:end),yy3(1:1:end),'-o','linewidth',2,'Markersize',12);
col4=get(h4,'color');
set(h4,'MarkerFaceColor',col4);

h5=semilogx(x4(1:1:end),yy4(1:1:end),'-o','linewidth',2,'Markersize',12);
col5=get(h5,'color');
set(h5,'MarkerFaceColor',col5);

semilogx([1 1e6],[y_fci,y_fci],'g--','linewidth',2)
xlim([1 1e6])

set(gca,'fontsize',20)
set(gca,'linewidth',2)

legend('CISD','QiankunNet','QiankunNet+CISD','QiankunNet+USCI','QiankunNet+USCI(p=0.01)')
legend('boxoff')

legend('boxoff')

xlabel('Iterations','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')

subplot(1,2,2)
x_sq = [1,1e5,1e5,1];   % 逆时针遍历每个点的x值
y_sq = [1e-10,1e-10,1.6e-3,1.6e-3];   % 逆时针遍历每个点的y值
fill(x_sq,y_sq,'p');   % 填充函数
hold on

h6=loglog([1 1e6],[y_cisd-y_fci,y_cisd-y_fci],'--','color',col1,'linewidth',2);
hold on

h7=loglog(x1(1:1:end),yy1(1:1:end)-y_fci,'-','linewidth',4,'color',col2,'Markersize',5);
set(h7,'MarkerFaceColor',col2);

h8=loglog(x2(1:1:end),yy2(1:1:end)-y_fci,'-','linewidth',4,'color',col3,'Markersize',5);
set(h8,'MarkerFaceColor',col3);
%h12.Color(4)=0.8;

h9=loglog(x3(1:1:end),yy3(1:1:end)-y_fci,'-','linewidth',4,'color',col4,'Markersize',5);
set(h9,'MarkerFaceColor',col4);
%h13.Color(4)=0.8;

h10=loglog(x4(1:1:end),yy4(1:1:end)-y_fci,'-','linewidth',4,'color',col5,'Markersize',5);
set(h10,'MarkerFaceColor',col5);

xlim([1 1e5])

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iterations','interpreter','latex')
ylabel('Absolute Energy error(Ha)','interpreter','latex')
