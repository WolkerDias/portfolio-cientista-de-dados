# *Dashboard* de Monitoramento das Estratégias de Retenção de Clientes

Este *dashboard* interativo foi desenvolvido no <a href="https://lookerstudio.google.com/" target="_blank">Looker Studio</a> para a equipe de CRM do aplicativo **TôComFome**, com o objetivo de monitorar e ajustar as estratégias de retenção de clientes, utilizando análises detalhadas do comportamento de *churn*. Estruturado em três páginas, ele oferece uma visão clara e precisa do desempenho das ações de retenção e seus impactos ao longo do tempo.

1. **Visão Geral de *Insights* de *Churn*:**  
   A primeira página apresenta uma visão sintética dos principais indicadores demográficos e comportamentais dos clientes, destacando características críticas associadas ao risco de *churn*. Essa visão permite uma rápida contextualização e identificação dos segmentos mais suscetíveis, com base nos *insights* extraídos da Análise Exploratória de Dados (AED). Para mais informações sobre a análise detalhada, acesse <a href="./Minimizando_o_Churn_de_Clientes_com_Machine_Learning.html#analise-exploratoria-dos-dados" target="_blank"><strong>Análise Exploratória de Dados</strong></a>.

2. **Monitoramento do KPI *Churn* ao Longo do Tempo:**  
   A segunda página acompanha o KPI de *churn* ao longo do tempo e o valor do ciclo de vida do cliente (LTV) que ajudam a entender o impacto financeiro da retenção. Para que os ajustes nas estratégias sejam implementados de forma controlada, o grupo de teste foi expandido progressivamente, começando com 30% da base total e aumentando a cada mês até atingir 100%.

3. **Análise dos Resultados das Ações de Retenção:**  
   A última página aprofunda a análise dos resultados do teste A/B, onde são comparados os grupos de controle (sem intervenções) e de teste (clientes que receberam ações de retenção). Esta visualização permite acompanhar a evolução do *churn* em ambos os grupos, permitindo ajustes nas estratégias de forma contínua e baseada nos dados do período anterior. Para mais informações sobre a metodologia implementada, acesse <a href="./Minimizando_o_Churn_de_Clientes_com_Machine_Learning.html#testes-a-b" target="_blank"><strong>Teste A/B</strong></a>.

  > 📌 Nota: Estes dados foram gerados aleatoriamente para exemplificar o desempenho do modelo, das estratégias e da simulação do teste A/B. Devido à interatividade do *dashboard*, algumas filtragens podem não ser pertinentes.

---

````{div} full-width
<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe 
    src="https://lookerstudio.google.com/embed/reporting/7302d9de-0b20-46c6-9f0a-07745bb59f2d/page/0Z9CE" 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
    frameborder="0" allowfullscreen 
    sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
  </iframe>
</div>
````