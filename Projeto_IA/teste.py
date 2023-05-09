rede_bayesiana={'chuva':['grama_molhada','regador'],
                'regador':['grama_molhada'],
                'grama_molhada':[]}

tabela_regador_chuva= {(0,0):0.6,
                       (1,0):0.4,
                       (1,1):0.01,
                       (0,1):0.99}

tabela_chuva={1:0.2,
              0:0.8}

tabela_regador_chuva_grama={(0,0):[0,1],
                            (0,1):[0.8,0.2],
                            (1,0):[0.9,0.1],
                            (1,1):[0.99,0.01]}


