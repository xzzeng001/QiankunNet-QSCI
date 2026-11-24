figure


hf=-107.50006350121;
mp2=-107.66268955840;
cisd=-107.65035476253;
ccsd=-107.65971140992;
fci=-107.66399143042;

subplot(2,2,1)
h1=plot(x1,y1,'-o','linewidth',2,'Markersize',12);
col1=get(h1,'color');
set(h1,'MarkerFaceColor',col1);

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs.','interpreter','latex')
ylabel('Nr. Params.','interpreter','latex')

subplot(2,2,2)

h2=plot(x1,y2,'-o','linewidth',2,'Markersize',12);
col2=get(h2,'color');
set(h2,'MarkerFaceColor',col2);

hold on
plot([0 x1(end)],[hf hf],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)],[mp2 mp2],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)],[cisd cisd],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)],[ccsd ccsd],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)],[fci fci],'--','linewidth',2,'Markersize',12)

legend('USCI','HF','MP2','CISD','CCSD','FCI')
legend('boxoff')

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')


subplot(2,2,3)
h3=plot(x2,y3,'-o','linewidth',2,'Markersize',12);
col3=get(h3,'color');
set(h3,'MarkerFaceColor',col3);

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs.','interpreter','latex')
ylabel('Nr. Params.','interpreter','latex')

subplot(2,2,4)

h4=plot(x2,y4,'-o','linewidth',2,'Markersize',12);
col4=get(h4,'color');
set(h4,'MarkerFaceColor',col4);

hold on
plot([0 x2(end)],[hf hf],'--','linewidth',2,'Markersize',12)
plot([0 x2(end)],[mp2 mp2],'--','linewidth',2,'Markersize',12)
plot([0 x2(end)],[cisd cisd],'--','linewidth',2,'Markersize',12)
plot([0 x2(end)],[ccsd ccsd],'--','linewidth',2,'Markersize',12)
plot([0 x2(end)],[fci fci],'--','linewidth',2,'Markersize',12)

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')