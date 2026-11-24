figure

fci=-107.4374218411345794;
cisd=-107.08194999;
hf=-106.47984262;

nn1=length(y1);
x1=1:nn1;
nn2=length(y2);
x2=1:nn2;
nn3=length(y3);
x3=1:nn3;
nn4=length(y4);
x4=1:nn4;

subplot(1,2,1)
h1=semilogx([1 5e5],[hf,hf],'--','linewidth',2);
col1=get(h1,'color');

hold on
h2=semilogx([1 5e5],[cisd,cisd],'--','linewidth',2);
col2=get(h2,'color');

%h3=semilogx([1 1e6],[fci,fci],'--','linewidth',2);
%col3=get(h3,'color');


%h4=plot(x1(1:1:end),y1(1:1:end),'-o','linewidth',2,'Markersize',12);
%col4=get(h4,'color');
%h4.Color(4)=0.8;
%set(h4,'MarkerFaceColor',col4);

h5=semilogx(x2(1:1:end),y2(1:1:end),'-o','linewidth',2,'Markersize',12);
col5=get(h5,'color');
h5.Color(4)=0.6;
set(h5,'MarkerFaceColor',col5);

%h6=semilogx(x3(1:1:end),y3(1:1:end),'-o','linewidth',2,'Markersize',12);
%col6=get(h6,'color');
%h6.Color(4)=0.4;
%set(h6,'MarkerFaceColor',col6);

h7=semilogx(x4(1:1:end),y4(1:1:end),'b-o','linewidth',2,'Markersize',12);
col7=get(h7,'color');
%h7.Color(4)=0.2;
set(h7,'MarkerFaceColor',col7);

semilogx([1 5e5],[fci,fci],'g--','linewidth',2)
xlim([1 5e5])

set(gca,'fontsize',20)
set(gca,'linewidth',2)

legend('HF','CISD','QiankunNet','qinakunNet-UCC','FCI')
legend('boxoff')

xlabel('Iterations','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')

subplot(1,2,2)
x_sq = [1,5e5,5e5,1];   % 逆时针遍历每个点的x值
y_sq = [1e-15,1e-15,1.6e-3,1.6e-3];   % 逆时针遍历每个点的y值
fill(x_sq,y_sq,'p');   % 填充函数
hold on

h8=loglog([1 5e5],[hf-fci,hf-fci],'--','color',col1,'linewidth',2);
hold on

h9=loglog([1 5e5],[cisd-fci,cisd-fci],'--','color',col2,'linewidth',2);
%h10=loglog([1 1e6],[vqe-fci,vqe-fci],'--','color',col3,'linewidth',2);

%h11=loglog(x1(1:1:end),abs(y1(1:1:end)-fci),'-','linewidth',4,'color',col4,'Markersize',5);
%set(h11,'MarkerFaceColor',col4);
%h11.Color(4)=0.8;

h12=loglog(x2(1:1:end),abs(y2(1:1:end)-fci),'-','linewidth',4,'color',col5,'Markersize',5);
set(h12,'MarkerFaceColor',col5);
h12.Color(4)=0.8;

%h13=loglog(x3(1:1:end),abs(y3(1:1:end)-fci),'-','linewidth',4,'color',col6,'Markersize',5);
%set(h13,'MarkerFaceColor',col6);
%h13.Color(4)=0.8;

h14=loglog(x4(1:1:end),abs(y4(1:1:end)-fci),'-','linewidth',4,'color',col7,'Markersize',5);
set(h14,'MarkerFaceColor',col7);
%h14.Color(4)=0.4;

%loglog([1 1e6],[1.6e-3,1.6e-3],'g--','linewidth',2)

%ylim([1e-15 10])
xlim([1 5e5])

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iterations','interpreter','latex')
ylabel('Absolute Energy error(Ha)','interpreter','latex')
