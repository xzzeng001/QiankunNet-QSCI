figure


hf=-106.47984262283;
mp2=-109.02326014844;
cisd=-107.12685011026;
ccsd=-106.98760909682;
fci=-107.43844013207;

subplot(1,2,1)
h1=plot(x1,y1,'-o','linewidth',2,'Markersize',12);
col1=get(h1,'color');
set(h1,'MarkerFaceColor',col1);

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs.','interpreter','latex')
ylabel('Nr. Params.','interpreter','latex')

subplot(1,2,2)

h2=plot(x1,y2,'-o','linewidth',2,'Markersize',12);
col2=get(h2,'color');
set(h2,'MarkerFaceColor',col2);

hold on
plot([0 x1(end)+1],[hf hf],'--','linewidth',2,'Markersize',12)
%plot([0 x1(end)],[mp2 mp2],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)+1],[cisd cisd],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)+1],[ccsd ccsd],'--','linewidth',2,'Markersize',12)
plot([0 x1(end)+1],[fci fci],'--','linewidth',2,'Markersize',12)

legend('USCI','HF','CISD','CCSD','FCI')
legend('boxoff')

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Nr. Configs','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')


%{
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
%}