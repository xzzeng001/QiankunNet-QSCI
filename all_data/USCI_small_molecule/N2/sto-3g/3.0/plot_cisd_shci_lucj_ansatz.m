figure


hf=-106.47984262283;
cisd=-107.12685011026;
ccsd=-106.98760909682;
shci=-107.3603963752;
fci=-107.43844013207;

nn1=length(y1);
x1=1:nn1;
nn2=length(y2);
x2=1:nn2;
nn3=length(y3);
x3=1:nn3;
nn=1000;

yy1=smooth(y1);
yy2=smooth(y2);
yy3=smooth(y3);

subplot(1,2,1)
h1=plot(x1,yy1,'-o','linewidth',2,'Markersize',12);
col1=get(h1,'color');
set(h1,'MarkerFaceColor',col1);

hold on
h2=plot(x2,yy2,'-o','linewidth',2,'Markersize',12);
col2=get(h2,'color');
set(h2,'MarkerFaceColor',col2);

h3=plot(x3,yy3,'-o','linewidth',2,'Markersize',12);
col3=get(h3,'color');
set(h3,'MarkerFaceColor',col3);

h4=plot([1 nn],[hf hf],'--','linewidth',2,'Markersize',12);
col4=get(h4,'color');
h5=plot([1 nn],[cisd cisd],'--','linewidth',2,'Markersize',12);
col5=get(h5,'color');
h6=plot([1 nn],[ccsd ccsd],'--','linewidth',2,'Markersize',12);
col6=get(h6,'color');
h7=plot([1 nn],[shci shci],'--','linewidth',2,'Markersize',12);
col7=get(h7,'color');
h8=plot([1 nn],[fci fci],'--','linewidth',2,'Markersize',12);
col8=get(h8,'color');

legend('LUCJ','USCI-cisd','USCI-shci','HF','CISD','CCSD','SHCI','FCI')
legend('boxoff')

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iteration','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')


subplot(1,2,2)
h9=plot(x1,yy1-fci,'-o','Color',col1,'linewidth',2,'Markersize',12);
set(h9,'MarkerFaceColor',col1);

hold on
h10=plot(x2,yy2-fci,'-o','Color',col2,'linewidth',2,'Markersize',12);
set(h10,'MarkerFaceColor',col2);

h11=plot(x3,yy3-fci,'-o','Color',col3,'linewidth',2,'Markersize',12);
set(h11,'MarkerFaceColor',col3);

h12=plot([1 nn],[hf-fci hf-fci],'--','Color',col4,'linewidth',2,'Markersize',12);
h13=plot([1 nn],[cisd-fci cisd-fci],'--','Color',col5,'linewidth',2,'Markersize',12);
h14=plot([1 nn],[ccsd-fci ccsd-fci],'--','Color',col6,'linewidth',2,'Markersize',12);
h15=plot([1 nn],[shci-fci shci-fci],'--','Color',col7,'linewidth',2,'Markersize',12);

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iteration','interpreter','latex')
ylabel('Absolute energy error (Ha)','interpreter','latex')
