figure

y_scf=-5.275451852309;
y_ccsd=-5.419404907949;
y_fci=-5.422958432642;

nn=length(y1);
x1=1:nn;

%subplot(1,2,1)
h1=plot(x1,y1,'-o','linewidth',2,'Markersize',12);
col1=get(h1,'color');
set(h1,'MarkerFaceColor',col1);

hold on
h2=plot([1 nn],[y_scf,y_scf],'--','linewidth',2);
col2=get(h2,'color');
h3=plot([1 nn],[y_ccsd,y_ccsd],'--','linewidth',2);
col3=get(h3,'color');
h4=plot([1 nn],[y_fci,y_fci],'--','linewidth',2);
col4=get(h4,'color');

legend([h1,h2,h3,h4],{'VQE','HF','CCSD','FCI'})
legend('boxoff')

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iteration','interpreter','latex')
ylabel('Energy (Ha)','interpreter','latex')

%{
subplot(1,2,2)
h5=plot(x1,abs(y1-y_fci),'-o','linewidth',2,'Color',col1,'Markersize',12);
set(h5,'MarkerFaceColor',col1);

hold on
h5=plot(x1,abs(y_scf-y_fci),'-o','linewidth',2,'Color',col2,'Markersize',12);
set(h5,'MarkerFaceColor',col2);

h6=plot(x1,abs(y_ccsd-y_fci),'-o','linewidth',2,'Color',col3,'Markersize',12);
set(h6,'MarkerFaceColor',col3);

set(gca,'fontsize',20)
set(gca,'linewidth',2)

xlabel('Iteration','interpreter','latex')
ylabel('Absolute energy error(Ha)','interpreter','latex')
%}